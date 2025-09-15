function [value] = basePipelineClassificationParallelTest( tbl, options )
    % Data Prep function with arg blocks
    %
    % Copyright 2021 The MathWorks Inc.
    
    arguments
       tbl table
       options.Normalization(1,1) string = "zscore"
    end

    %Table and properties
    value = tbl;
    
    %categorical and response vars
    features = string(value.Properties.VariableNames(1:end-1));
    response_var = "Origin"; 

    value{:, features} = ones(height(value), width(value)-1);
    
    %Define Partition: Train/Test 
    value = baseml.partition( value );
    
    %Store Response, Features, and Partition 
    value = addprop(value, "Features", "table" );
    value = addprop(value, "Response", "table" );
    value = addprop(value, "TrainingObservations", "table" );
    value = addprop(value, "Options", "table" );
    
    value.Properties.CustomProperties.Features = features;
    value.Properties.CustomProperties.Response = response_var;
    value.Properties.CustomProperties.TrainingObservations = value.Partition == "Train";
    value.Properties.CustomProperties.Options = options;

end

