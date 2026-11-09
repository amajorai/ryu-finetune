use super::{normalize_remote_base_url, response_job_id, valid_base_model_id, valid_output_name};
use serde_json::json;

// ── OpenAPI document ───────────────────────────────────────────────────────

/// This app's own manifest, read at compile time. The route contract lives there,
/// so the invariants below compare the document against the real declaration
/// rather than against a second list that could drift from it.
fn openapi_manifest() -> serde_json::Value {
    serde_json::from_str(include_str!("../../../manifest.json")).expect("valid JSON")
}

/// The manifest sidecar whose HTTP surface this router serves: the one that
/// declares an `http.mount`. Selected BY mount rather than by index because an app
/// may declare a second, mountless sidecar (finetune already does), and
/// `sidecars[0]` would then quietly start asserting against the wrong process.
fn mounted_sidecar() -> serde_json::Value {
    openapi_manifest()["sidecars"]
        .as_array()
        .expect("sidecars must be an array")
        .iter()
        .find(|s| s["http"]["mount"].is_string())
        .expect("one sidecar must declare an http.mount")
        .clone()
}

/// A manifest route (relative to the mount, in axum's `:param` form) rewritten
/// into the form the OpenAPI document uses (absolute, in `{param}` form).
///
/// The two forms differ ON PURPOSE — the router registers paths relative to the
/// mount because Core nests it there, while the `#[utoipa::path]` annotations carry
/// the absolute EXTERNAL path a caller actually hits. Normalise here; do not
/// "align" either side.
fn doc_path_for(mount: &str, route: &str) -> String {
    let joined = if route == "/" {
        mount.to_owned()
    } else {
        format!("{mount}{route}")
    };
    joined
        .split('/')
        .map(|seg| match seg.strip_prefix(':') {
            Some(name) => format!("{{{name}}}"),
            None => seg.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[test]
fn openapi_doc_is_served_and_non_empty() {
    // The doc is no longer dead code: Core fetches it to derive tools.
    assert!(!super::openapi().paths.paths.is_empty());
}

#[test]
fn every_declared_route_appears_in_the_openapi_doc() {
    // The direction that decides tool yield. Core's `ext_api::lower` keeps only the
    // document operations the manifest ALSO declares, so a declared route with no
    // `#[utoipa::path]` annotation is a tool that silently never exists — nothing
    // errors, the agent simply cannot call it. (The other direction is harmless: an
    // annotated path the manifest does not declare is dropped by the same filter.)
    let sidecar = mounted_sidecar();
    let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
    let doc = super::openapi();
    for route in sidecar["http"]["routes"]
        .as_array()
        .expect("routes must be an array")
    {
        let path = route["path"].as_str().expect("a route path");
        let expected = doc_path_for(mount, path);
        assert!(
            doc.paths.paths.contains_key(&expected),
            "'{path}' is declared in manifest.json but the OpenAPI document has no \
             '{expected}' operation — Core derives no tool for it"
        );
    }
}

// ── Request-body schemas ───────────────────────────────────────────────────

/// The one pointer Core reads to give a derived write tool its arguments.
fn body_schema(wire: &serde_json::Value, path: &str, method: &str) -> serde_json::Value {
    wire.pointer(&format!(
        "/paths/{}/{method}/requestBody/content/application~1json/schema",
        path.replace('/', "~1")
    ))
    .unwrap_or_else(|| panic!("{method} {path} must declare a JSON request body"))
    .clone()
}

#[test]
fn post_routes_document_their_request_body() {
    // The regression this locks down: both annotations here used to say
    // `request_body = serde_json::Value`, which serialises to an untyped schema.
    // Core derives a tool per operation and fills `input_schema` from THIS node,
    // so an untyped body produced a tool the model could discover, could call,
    // and could never pass a single argument to — discoverable and useless, with
    // nothing logged to explain it. Training is the most expensive thing this
    // node does; an agent that cannot name a base model cannot start one.
    //
    // A `$ref` is the CORRECT and expected shape, not a near-miss: Core's
    // `openapi_import::resolve_ref` resolves it against `components.schemas`
    // before reading `properties`. So accept either a ref or inlined properties;
    // asserting "inlined" would fail on a healthy document.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    for (path, method) in [
        ("/api/finetune/start", "post"), // start -> StartJobBody
        ("/api/finetune/merge", "post"), // merge -> MergeBody
    ] {
        let schema = body_schema(&wire, path, method);
        assert!(
            schema.get("$ref").is_some() || schema.get("properties").is_some(),
            "a derived write tool for {method} {path} would have no arguments: {schema}"
        );
    }
}

#[test]
fn every_request_body_ref_resolves_against_components() {
    // The half of the retrofit that a `$ref`-shaped assertion alone cannot see:
    // a `$ref` pointing at a schema that was never registered in
    // `components(schemas(...))` looks identical in the operation and still
    // yields zero arguments once Core tries to resolve it. Walk every request
    // body in the document and check the target actually exists and carries
    // properties.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    let paths = wire["paths"].as_object().expect("paths must be an object");
    let mut checked = 0usize;
    for (path, item) in paths {
        for (method, op) in item.as_object().expect("a path item is an object") {
            let Some(schema) = op.pointer("/requestBody/content/application~1json/schema") else {
                continue;
            };
            let Some(reference) = schema.get("$ref").and_then(|r| r.as_str()) else {
                // Inlined schemas are fine as long as they describe something.
                // The failure this catches in practice is `request_body =
                // Option<T>`, which utoipa renders as a nullable `oneOf` wrapper:
                // Core resolves only a TOP-LEVEL `$ref`, so the wrapper reaches the
                // importer unresolved and contributes no properties at all.
                assert!(
                    schema.get("properties").is_some(),
                    "{method} {path} has a request-body schema Core cannot read \
                     (a `oneOf` here means `request_body = Option<T>` — use the \
                     plain type): {schema}"
                );
                checked += 1;
                continue;
            };
            let name = reference
                .strip_prefix("#/components/schemas/")
                .unwrap_or_else(|| panic!("unexpected ref form '{reference}' at {method} {path}"));
            let target = wire
                .pointer(&format!("/components/schemas/{name}"))
                .unwrap_or_else(|| {
                    panic!(
                        "{method} {path} refs '{name}' but it is missing from \
                         components.schemas — add it to components(schemas(..))"
                    )
                });
            assert!(
                target.get("properties").is_some(),
                "{method} {path} refs '{name}', which has no properties: {target}"
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked, 2,
        "expected both write routes to carry a body schema, saw {checked}"
    );
}

#[test]
fn a_nested_struct_argument_is_self_describing() {
    // `StartJobBody::training` is an `Option<TrainingSpec>`. utoipa wraps that in
    // `oneOf: [null, …]`, and Core resolves a `$ref` only at the TOP of a node —
    // so a ref nested inside the wrapper would reach the model as an opaque
    // pointer. `#[schema(inline)]` is what makes the real hyper-parameters
    // visible; this test fails the moment someone removes it.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    let training = wire
        .pointer("/components/schemas/StartJobBody/properties/training")
        .expect("StartJobBody must document `training`");
    let variants = training["oneOf"]
        .as_array()
        .expect("an optional struct field is a oneOf wrapper");
    let object = variants
        .iter()
        .find(|v| v["type"] == "object")
        .expect("the non-null variant must be an inlined object, not a $ref");
    for field in ["epochs", "learning_rate", "max_steps", "load_in_4bit"] {
        assert!(
            object["properties"].get(field).is_some(),
            "TrainingSpec::{field} is invisible to the model: {object}"
        );
    }
}

#[test]
fn a_required_struct_argument_stays_resolvable() {
    // `dataset` is NOT optional, so utoipa emits a bare `$ref` at the top of the
    // property node — the one shape Core's one-level nested resolution does
    // follow. No `#[schema(inline)]` needed; what IS needed is the component
    // entry it points at, which this asserts end to end.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    let reference = wire
        .pointer("/components/schemas/StartJobBody/properties/dataset/$ref")
        .and_then(|r| r.as_str())
        .expect("`dataset` must be a top-level $ref Core can follow");
    assert_eq!(reference, "#/components/schemas/DatasetSpec");
    assert!(
        wire.pointer("/components/schemas/DatasetSpec/properties/samples")
            .is_some(),
        "DatasetSpec is not registered, so the ref dangles and `dataset` is opaque"
    );
    // `samples` is a `Vec<serde_json::Value>`. utoipa inlines `Value` as an
    // any-schema; if it ever emitted a NAMED component instead, the ref would sit
    // two hops deep (property → items) — past the single level Core expands
    // inside an already-resolved object — and reach the model as an opaque
    // pointer that every other assertion here would still pass.
    assert!(
        wire.pointer("/components/schemas/DatasetSpec/properties/samples/items/$ref")
            .is_none(),
        "`samples` items became a $ref Core will not expand"
    );
}

#[test]
fn schema_descriptions_carry_no_internal_rationale() {
    // utoipa lifts a STRUCT's doc comment into the schema's own `description`,
    // exactly as it lifts field docs into property descriptions — so a `///`
    // paragraph explaining why a type is not the axum extractor would ship to the
    // model as part of the tool. The convention that prevents it: one `///` line
    // naming the body, and every rationale paragraph below it demoted to `//`.
    // Wrapped prose is fine — the tell is VOCABULARY, so this greps for the Rust
    // implementation words that only ever appear in rationale, never in something
    // written for a caller.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    let schemas = wire["components"]["schemas"]
        .as_object()
        .expect("components.schemas must be an object");
    for (name, schema) in schemas {
        let mut descriptions = vec![schema.get("description")];
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            descriptions.extend(props.values().map(|p| p.get("description")));
        }
        for description in descriptions
            .into_iter()
            .flatten()
            .filter_map(|d| d.as_str())
        {
            for leak in ["axum", "utoipa", "extractor", "Deserialize", "serde_json"] {
                assert!(
                    !description.contains(leak),
                    "{name} ships the word '{leak}' to the model in a schema \
                     description — demote that rationale from `///` to `//`: \
                     {description:?}"
                );
            }
        }
    }
}

#[test]
fn local_capability_requires_a_ready_worker() {
    let hardware = (true, String::new());
    assert_eq!(
        super::effective_local_capability(
            hardware.clone(),
            Some(&serde_json::json!({
                "can_finetune": true,
                "reason": ""
            }))
        ),
        (true, String::new())
    );

    let (ready, reason) = super::effective_local_capability(
        hardware.clone(),
        Some(&serde_json::json!({
            "can_finetune": false,
            "reason": "unsloth is not installed"
        })),
    );
    assert!(!ready);
    assert!(!reason.contains("unsloth is not installed"));
    assert_eq!(reason, super::worker_reason::WORKER_CANNOT_TRAIN_MESSAGE);

    let (ready, reason) = super::effective_local_capability(hardware, None);
    assert!(!ready);
    assert!(reason.contains("worker is unavailable"));
}

#[test]
fn body_field_docs_reach_the_schema_as_argument_descriptions() {
    // Doc comments on the body-struct fields are the whole payoff of the
    // retrofit: they are the only prose the model reads when choosing arguments.
    // utoipa lifts them into `description`, so a future edit that drops them
    // silently degrades tool-call quality with no compile error.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    let base = wire
        .pointer("/components/schemas/StartJobBody/properties/base_model_id/description")
        .and_then(|d| d.as_str())
        .unwrap_or_default();
    assert!(
        base.contains("Hugging Face repo id"),
        "StartJobBody::base_model_id lost its doc comment, got {base:?}"
    );
}

#[test]
fn only_the_body_carrying_routes_declare_a_request_body() {
    // The other direction of the same bug. `capability`/`list`/`adapters`, the
    // single-job read, the SSE stream, and the DELETE cancel take no JSON body at
    // all — their handlers have no `Json` extractor. Declaring one would document
    // something the endpoint never reads, and (before the retrofit) an untyped one
    // at that.
    let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
    for (path, method) in [
        ("/api/finetune/capability", "get"),
        ("/api/finetune/list", "get"),
        ("/api/finetune/adapters", "get"),
        ("/api/finetune/{id}", "get"),
        ("/api/finetune/{id}", "delete"),
        ("/api/finetune/{id}/stream", "get"),
    ] {
        let op = wire
            .pointer(&format!("/paths/{}/{method}", path.replace('/', "~1")))
            .unwrap_or_else(|| panic!("{method} {path} must have an operation"));
        assert!(
            op.get("requestBody").is_none(),
            "{method} {path} takes no body but the document declares one"
        );
    }
    // …and the id the per-job routes DO take must still be an argument.
    for path in ["/api/finetune/{id}", "/api/finetune/{id}/stream"] {
        let op = wire
            .pointer(&format!("/paths/{}/get", path.replace('/', "~1")))
            .expect("a GET operation");
        assert!(
            op.get("parameters").is_some(),
            "{path} must still document its path id"
        );
    }
}

#[test]
fn remote_targets_reject_private_hosts_and_url_smuggling() {
    assert!(normalize_remote_base_url("https://127.0.0.1").is_err());
    assert!(normalize_remote_base_url("https://169.254.169.254").is_err());
    assert!(normalize_remote_base_url("https://gpu.example.test/path").is_err());
    assert!(normalize_remote_base_url("https://user:pass@gpu.example.test").is_err());
    assert!(normalize_remote_base_url("https://gpu.example.test/?token=secret").is_err());
    assert_eq!(
        normalize_remote_base_url("https://GPU.Example.Test:8443/").unwrap(),
        "https://gpu.example.test:8443"
    );
}

#[test]
fn worker_response_must_contain_a_bounded_job_id() {
    assert_eq!(
        response_job_id(&json!({ "job_id": "job-1" })).unwrap(),
        "job-1"
    );
    assert!(response_job_id(&json!({})).is_err());
    assert!(response_job_id(&json!({ "job_id": "   " })).is_err());
    assert!(response_job_id(&json!({ "job_id": "x".repeat(129) })).is_err());
    for id in [
        "../etc/passwd",
        "job?redirect=1",
        "job/child",
        "job%2Fchild",
        ".",
        "..",
    ] {
        assert!(
            response_job_id(&json!({ "job_id": id })).is_err(),
            "path-significant job id must be rejected: {id}"
        );
    }
}

#[test]
fn model_and_output_names_cannot_become_loader_or_path_inputs() {
    assert!(valid_base_model_id("unsloth/llama-3-8b"));
    for id in [
        "/etc/passwd",
        "file:///tmp/model",
        "../model",
        "org/model/extra",
    ] {
        assert!(!valid_base_model_id(id), "unsafe model id accepted: {id}");
    }
    assert!(valid_output_name("adapter-v1"));
    for name in ["../adapter", ".", "adapter/name", "adapter\0name"] {
        assert!(
            !valid_output_name(name),
            "unsafe output name accepted: {name:?}"
        );
    }
}

// ── Worker refusals ────────────────────────────────────────────────────────

/// The Companion shows the `error` string and its localizer matches it whole, so
/// a worker refusal must reach it as the worker's own sentence: no route, status
/// or "unsloth … failed" prefix.
#[test]
fn worker_refusals_relay_the_plain_sentence() {
    use super::{
        worker_failure, WorkerRefusal, WORKER_REFUSAL_MESSAGE, WORKER_UNAVAILABLE_MESSAGE,
    };

    let refusal = WorkerRefusal::from_body(
        reqwest::StatusCode::BAD_REQUEST,
        &json!({ "error": " dataset has no samples " }),
        "/finetune",
    );
    let (status, body) = worker_failure(&anyhow::Error::from(refusal));
    assert_eq!(status, axum::http::StatusCode::BAD_GATEWAY);
    assert_eq!(body, json!({ "error": "dataset has no samples" }));

    let silent = WorkerRefusal::from_body(
        reqwest::StatusCode::UNPROCESSABLE_ENTITY,
        &json!({ "detail": [] }),
        "/finetune/merge",
    );
    let (_, body) = worker_failure(&anyhow::Error::from(silent).context("merging"));
    assert_eq!(body, json!({ "error": WORKER_REFUSAL_MESSAGE }));

    let (_, body) = worker_failure(&anyhow::anyhow!("error sending request for url"));
    assert_eq!(body["error"], WORKER_UNAVAILABLE_MESSAGE);
    assert!(body["hint"].is_string());
}
