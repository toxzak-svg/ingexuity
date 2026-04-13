# ============================================================================
# Intelligence.jl — Output Layer: learned patterns from prediction outcomes
# ============================================================================
module Intelligence

using ...Types

"""Record a prediction outcome and update intelligence"""
function record!(
    intel::Intelligence,
    prediction::Prediction,
    was_correct::Bool
)::Intelligence
    new_correct = intel.correct_predictions + (was_correct ? 1 : 0)
    new_total = intel.total_predictions + 1
    new_accuracy = new_correct / new_total

    Intelligence(new_correct, new_total, new_accuracy, now())
end

"""Get intelligence as a summary string"""
function summary(intel::Intelligence)::String
    "Intelligence: $(intel.correct_predictions)/$(intel.total_predictions) predictions correct " *
    "($(round(intel.accuracy * 100; digits=1))% accuracy)"
end

end # module
