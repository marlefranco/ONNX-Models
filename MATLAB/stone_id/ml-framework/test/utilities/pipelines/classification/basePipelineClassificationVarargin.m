function [value] = basePipelineClassificationVarargin( tbl, varargin )
    % Data Prep function with arg blocks
    %
    % Copyright 2021 The MathWorks Inc.
    
    if nargin == 1
        norm = "zscore";
    else
        norm = varargin{2};
    end

    %Table and properties
    value = tbl;
    
    %categorical and response vars
    response_var = "Origin"; 
    
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
    
    %Normalize
    feature_num  = normalize(feature_num, norm);
    
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
    
    value.Properties.CustomProperties.Features = features;
    value.Properties.CustomProperties.Response = response_var;

end

