function value = basePipelinePredict( mdl, data, options )

arguments
    mdl
    data {mustBeA(data,["table","timetable"])}
    options.Response (1,1) string = "Response"
    options.Normalization (1,1) string {mustBeMember(options.Normalization, ...
        ["none","zscore", "range"])} = "zscore"
    options.NormalizationInfo table = table()
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
%[value, featurenames] = util.rmconstant( value, "DataVariables", featurenames );

featurenames = intersect(value.Properties.VariableNames, featurenames, "stable");

if options.FeatureExtraction == "descriptivestatistics"
    [value, featurenames] = ops.util.getBasicStats(value, ...
        "FeatureNames", featurenames, ...
        "ResponseName", responsename);
end

%Normalize
info = options.NormalizationInfo;
switch options.Normalization
    case {'zscore', 'range'}
        value = util.scaler( value, info );
    case 'none'
        %Do nothing
    otherwise
        error( "Unhandled feature transformation" )
        
end %options.Normalization

switch options.FeatureSelection
    case "pca"
        value = util.dimensionreduce( value, "pca", ...
            "PredictorNames", featurenames);
         value = movevars(value, responsename, 'After', ...
             value.Properties.VariableNames(end));
         
    case "fscchi2"
        
        try
            % select features from training data
            value = value (:, [mdl.ExpandedPredictorNames, responsename]);
        catch
            % select features from training data
            value = value (:, [mdl.PredictorNames, responsename]);
        end
        
    case "tsneWsparseFilt"
        
        numFeatures = 100;
        response = value{:,end};
        features = value{:,1:end-1};
        Mdl = sparsefilt(features, numFeatures);
        transFeat = transform(Mdl, features);
        Y = tsne(transFeat,'Algorithm','exact','Distance','cosine');
        
        value = array2table(Y);
        value.(responsename) = response;
        
    case "tsneWrica"
        
        numFeatures = 100;
        response = value{:,end};
        features = value{:,1:end-1};
        Mdl = rica(features, numFeatures);
        transFeat = transform(Mdl, features);
        Y = tsne(transFeat,'Algorithm','exact','Distance','cosine');
        
        value = array2table(Y);
        value.(responsename) = response;
end

%Define Partition a holdout/test set( specify cross validation in later step )
% value = baseml.partition( value );
% value = value(value.Partition == "Test", :);

%Convert response/target to categorical (If Classification).
value.(responsename) = categorical( value.(responsename) );

%Append Prediction
[prediction, scores] = baseml.predict(value, mdl);
value.Prediction = prediction;

try
    classNames = string(mdl.ClassNames);
catch
    classNames = string(mdl.Classes);
end

scoresNorm = arrayfun(@(x) exp(scores(x,:))./sum(exp(scores(x,:))), ...
    1:size(scores,1), 'UniformOutput', false);
scoresNorm = vertcat(scoresNorm{:});
scoresT = array2table(scoresNorm, 'VariableNames', classNames);

value = horzcat(value, scoresT);
value = mergevars(value, classNames, ...
    'NewVariableName', 'Scores', 'MergeAsTable', true);

end %function
