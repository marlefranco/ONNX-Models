function [value] = basePipelineRegression( tbl, options )
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
    response_var = "MPG"; 
    
    %Remove NaN's from response var
    value = rmmissing(value);
    
    %Extract features and response
    features = value(:,1:end-1);
    response = value(:, response_var);
    
    %Extract numeric predictors
    tFNum = varfun(@(x)isnumeric(x), features, "OutputFormat", "uniform");
    feature_num = features(:, tFNum);
    
    %Remove constant vars
    tFUni = varfun(@(x)all(x == x(1)), feature_num, "OutputF", "uni");
    feature_num = feature_num(:, ~tFUni);
     
    %Convert text based features to categorical 
    value = util.text2categorical( value );
    
    %Normalize
    feature_num  = normalize(feature_num, options.Normalization);
    response  = normalize(response, options.Normalization);
    
    if ~any(~tFNum)
        features = feature_num;
    else
        features = [feature_num, value(:,~tFNum)];
    end
    
    value = [features, response];
    
    %Define Response and Features
    features  = string(features.Properties.VariableNames);
    
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

