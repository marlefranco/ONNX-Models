function [cm, result] = CMprediction(model, data)

result = basePipelinePredict( model.Results.mdl, data, model.Results.pipelinesweep{:});
accuracy = sum(result.("Response") == result.Prediction) / numel(result.("Response"));
f = figure();
cm = viz.confusionchart( result, "Response", "Prediction", ...       
    "Title", "Test Data Accuracy = " + 100*accuracy + "%", ...       
    "Normalization", "row-normalized", ...    
    "RowSummary","row-normalized","ColumnSummary","column-normalized");
f.Position = [10,10,1000,700];
end