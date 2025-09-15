function value = basePipelineRULParallelTest( tbl, options )
% pdmTrainBasePipeline Prepare/process data for training on RUL (base case)
%
% Description:
%   Extract features using Diagnostic Feature Designer 
%   Smooth generated features
%   Define Partition
%   Rank features
%   Normalization and Dimensionality Reduction using PCA
%   Map/Define health indicator

% Copyright 2021 The MathWorks Inc.

    arguments
       tbl table
       options.Normalization (1,1) string {mustBeMember(options.Normalization, ["zscore", "center", "range", "scale", "norm"])} = "zscore"
    end

    %Generate features
    normFunc = @(x) {normalize(x,options.Normalization,'DataVariables','Condition')};
    
    tbl = rowfun(normFunc, tbl, 'ExtractCellContents', true);

    data = tbl.Var1;

    for ii = 1:numel(data)
        data{ii}.Time = ones(height(data{ii}),1);
    end

    tbl.Var1 = data;
    
    value = tbl;
    value.Properties.VariableNames = "HealthIndicator";
    
    value = addprop(value, "DataVariable", "table");
    value = addprop(value, "LifeTimeVariable", "table");
    value = addprop(value, "HealthIndicatorName", "table");
    value = addprop(value, "Options", "table" );
    
    value.Properties.CustomProperties.DataVariable = "Condition";
    value.Properties.CustomProperties.LifeTimeVariable = "Time";
    value.Properties.CustomProperties.HealthIndicatorName = "HealthIndicator";
    value.Properties.CustomProperties.Options = options;
    
             
end %pdmTrainBasePipeline