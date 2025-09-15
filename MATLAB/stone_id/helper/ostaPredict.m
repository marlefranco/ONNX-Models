function value = ostaPredict( mdl, data, options )

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
    options.Class (1,1) string {mustBeMember(options.Class, ...
        ["AllClass", "2-Class", "4-Class"])} = "AllClass"
    
end %arguments

%Place your data preparation code here.
value = data;

if options.Class == "2-Class"
    value.Response = mergecats(value.Response, ["ClassA", "ClassB", "ClassC"], "ClassABC");
    value.Response = mergecats(value.Response, ["ClassD1", "ClassD2"], "ClassD");
elseif options.Class == "4-Class"
    value.Response = mergecats(value.Response, ["ClassD1", "ClassD2"], "ClassD");
end

featurenames = mdl.PredictorNames;
responsename = options.Response;

%Clean/Remove constant vars
value = util.rmconstant( value, "DataVariables", featurenames );

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

%Convert response/target to categorical (If Classification).
value.(responsename) = categorical( value.(responsename) );

%Append Prediction
prediction = baseml.predict(value, mdl);
value.Prediction = prediction;

end %function
