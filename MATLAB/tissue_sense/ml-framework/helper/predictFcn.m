function result = predictFcn(filename, data)
% predictFcn Prediction function for OSTA data
%
% Syntax:
%   result = predictFcn(filename, data)

[mdl, pipesettings] = loadExperimentItem(filename);

result = basePipelinePredict( mdl, data, pipesettings{:} );

end