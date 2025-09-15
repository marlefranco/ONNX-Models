function value = basePipelineTrain( data, options )

arguments
    data {mustBeA(data,["table","timetable"])}
    options.Response (1,1) string = "Response"
    options.Normalization (1,1) string {mustBeMember(options.Normalization, ...
        ["none","zscore", "range"])} = "zscore"
    options.FeatureExtraction (1,1) string {mustBeMember(options.FeatureExtraction, ...
        ["none", "descriptivestatistics"])} = "none"
    options.FeatureSelection (1,1) string {mustBeMember(options.FeatureSelection, ...
        ["none", "fscchi2", "pca", "tsneWsparseFilt", "tsneWrica"])} = "fscchi2"
    
end %arguments

%Common data preparation steps for supervised training
%   - General Cleaning: Remove/impute instances with missing data,
%     constant features, ...
%   - Feature Engineering: Optionally transform, derive, buffer
%     and aggregate features,...
%   - Feature Scaling: If feature magnitudes differ, apply range
%     scaling, zscore normalization,...
%   - Feature Selection: Apply data dependent feature selection methods


%General Notes:
%   - The toolbox provides convenience utilites for data preparation and ML.
%     Please see 'util' and 'baseml' for additional details.
%   - See demos for examples of completed pipelines.
%   - The relative sequence of steps may vary. Consider creating separate
%     pipelines for structural changes.
%   - Ensure pipeline defines variables assigned in custom properties including
%     a list of: featurenames and responsename (as strings), partition info (as logical),
%     and pipeline settings (e.g. input arguments for provenance/reproduction ).

%Place your data preparation code here.
value = data;

featurenames = value.Properties.VariableNames(1:end-1);
responsename = options.Response;

%Clean/Remove constant vars
[value, featurenames] = util.rmconstant( value, "DataVariables", featurenames );

featurenames = intersect(value.Properties.VariableNames, featurenames, "stable");

if options.FeatureExtraction == "descriptivestatistics"
    [value, featurenames] = ops.util.getBasicStats(value, ...
        "FeatureNames", featurenames, ...
        "ResponseName", responsename);
end

%Normalize
if options.Normalization ~= "none"
    
    [value, info] = util.normalize(value, options.Normalization, ...
        "DataVariables", featurenames);
    
    options.NormalizationInfo = info;
    
end %options.Normalization

switch options.FeatureSelection
    case "pca"
        [value, ~, dr] = util.dimensionreduce( value, "pca", ...
            "PredictorNames", featurenames);
         value = movevars(value, 'Response', 'After', value.Properties.VariableNames(end));
        
         % extract features that contribute to 95% variance
         idx = find(cumsum(dr.explained) > 95);
         
         featurenames = value.Properties.VariableNames(1:min(idx(1), 10));
         value = value(:,[string(featurenames) responsename]);
         
    case "fscchi2"
        
        [value, featurenames] = util.featureselection( value, "fscchi2", ...
            "PredictorNames", featurenames, ...
            "ResponseName", responsename);
        
    case "tsneWsparseFilt"
        
        numFeatures = 100;
        response = value{:,end};
        features = value{:,1:end-1};
        Mdl = sparsefilt(features, numFeatures);
        transFeat = transform(Mdl, features);
        Y = tsne(transFeat,'Algorithm','exact','Distance','cosine');
        
        value = array2table(Y);
        featurenames = value.Properties.VariableNames;
        value.(responsename) = response;
        
    case "tsneWrica"
        
        numFeatures = 100;
        response = value{:,end};
        features = value{:,1:end-1};
        Mdl = rica(features, numFeatures);
        transFeat = transform(Mdl, features);
        Y = tsne(transFeat,'Algorithm','exact','Distance','cosine');
        
        value = array2table(Y);
        featurenames = value.Properties.VariableNames;
        value.(responsename) = response;
end

%Convert response/target to categorical (If Classification).
value.(responsename) = categorical( value.(responsename) );

%Define Partition a holdout/test set( specify cross validation in later step )
% validation set
value = baseml.partition( value );

%Store Response, Features, Partition, and Pipeline options.
value = addprop(value, "Features", "table" );
value = addprop(value, "Response", "table" );
value = addprop(value, "TrainingObservations", "table" );
value = addprop(value, "Options", "table" );

value.Properties.CustomProperties.Features = featurenames;
value.Properties.CustomProperties.Response = responsename;
value.Properties.CustomProperties.TrainingObservations = value.Partition == "Train";
value.Properties.CustomProperties.Options = options;

end %function
