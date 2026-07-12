use ingexuity_core::{process_turn, ConversationState, HeuristicBackend};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u64,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    message: String,
    expected_fragment: String,
}

#[test]
fn synthetic_baseline_is_deterministic() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../fixtures/synthetic/baseline.json"
    ))
    .expect("baseline fixture must remain valid JSON");

    assert_eq!(fixture.schema_version, 1);
    let backend = HeuristicBackend;

    for case in fixture.cases {
        let mut first = ConversationState::new(Uuid::nil());
        let mut second = ConversationState::new(Uuid::nil());

        let first_output = process_turn(&mut first, &case.message, &backend)
            .unwrap_or_else(|error| panic!("case {} failed: {error}", case.id));
        let second_output = process_turn(&mut second, &case.message, &backend)
            .unwrap_or_else(|error| panic!("case {} failed on replay: {error}", case.id));

        assert!(
            first_output.text.contains(&case.expected_fragment),
            "case {} output did not contain {:?}: {:?}",
            case.id,
            case.expected_fragment,
            first_output.text
        );
        assert_eq!(first_output.text, second_output.text, "case {}", case.id);
        assert_eq!(first.user_model, second.user_model, "case {}", case.id);
    }
}
