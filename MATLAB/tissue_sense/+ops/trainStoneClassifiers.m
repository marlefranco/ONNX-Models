function [model1, model2] = trainStoneClassifiers(dataIn, integrationTime)

arguments
    dataIn (:,:) table
    integrationTime (1,1) double
end

% consider only current integration time
dataIn(dataIn.IntegrationTimeUsed ~= string(integrationTime) ,:) = [];

% split into test and train
rng(0);
trainPercent = 80; % percentage to use for training
splitn = splitlabels(dataIn.Response, trainPercent/100, 'random');

predictorNames = string(dataIn.Properties.VariableNames(...
                contains(dataIn.Properties.VariableNames, "Feature")))';
data = dataIn(:,[predictorNames; "Response"]);
%% Stone versus Non-stone classifiers

dataModel1 = data;
dataModel1.Response(ismember(dataModel1.Response, ["Access Sheath 13/15", ...
    "Endoscope", "Tissue-Calyx", "Tissue-Ureter"])) = "Non-stone";
dataModel1.Response(ismember(dataModel1.Response, ["BEGO", "COM", "UA"])) = "Stone";

[results1, info1, session1] = ops.trainAndChooseBestModel(dataModel1, splitn);
model1.Results = results1;
model1.Info = info1;
model1.Session = session1;

%% Train Stone classifiers

dataModel2 = data;
dataModel2.Partition = repmat("Train", height(dataModel2), 1);
dataModel2.Partition(splitn{2}) = "Test";

isNonStone = ismember(dataModel2.Response, ["Access Sheath 13/15", ...
    "Endoscope", "Tissue-Calyx", "Tissue-Ureter"]);
dataModel2(isNonStone,:) = [];

splitn{1} = find(dataModel2.Partition=="Train");
splitn{2} = find(dataModel2.Partition=="Test");
dataModel2.Partition = [];

[results2, info2, session2] = ops.trainAndChooseBestModel(dataModel2, splitn);
model2.Results = results2;
model2.Info = info2;
model2.Session = session2;

end