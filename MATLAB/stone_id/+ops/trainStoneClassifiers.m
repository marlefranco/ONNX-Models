function [model1, model2] = trainStoneClassifiers(dataIn, integrationTime)
%function [model1, model2] = trainStoneClassifiers(dataIn, integrationTime, splitn)
arguments
    dataIn (:,:) table
    integrationTime (1,1) double
   %splitn (1,2) cell
end

% consider only current integration time
dataIn(dataIn.IntegrationTimeUsed ~= string(integrationTime) ,:) = [];
dataInt = dataIn;
currentFilter = dataIn.ResponseRegression >= 2 & dataIn.ResponseRegression < 5;
dataIn = dataIn(currentFilter, :);
% % split into test and train
rng(0);
trainPercent = 80; % percentage to use for training
% splitn = splitlabels(dataIn.Response, trainPercent/100, 'random');

% if isempty(splitn)
splitn = ops.partitionDataBasedonFilenames(dataIn, trainPercent);
% end

% if nargin < 3
% splitn = ops.partitionDataBasedonFilenames(dataIn, trainPercent);
% end

predictorNames = string(dataIn.Properties.VariableNames(...
    contains(dataIn.Properties.VariableNames, "Feature")))';
data = dataIn(:,[predictorNames; "Response"]);

%% Stone versus Non-stone classifiers

dataModel1 = data;
dataModel1.Response(contains(dataModel1.Response, ["Access Sheath", ...
    "Endoscope", "Tissue-Calyx", "Tissue-Ureter", "BEGO", "Guidewire"])) = "Non-stone";
dataModel1.Response(ismember(dataModel1.Response, ["COM", "UA"])) = "Stone";

% check if data is not empty and if it contains at least 2 classes
if ~isempty(dataModel1) && length(unique(dataModel1.Response))>1

    [results1, info1, session1] = ops.trainAndChooseBestModel(dataModel1, splitn);
    model1.Results = results1;
    model1.Info = info1;
    model1.Session = session1;
    model1.dataTrain = dataModel1(splitn{1}, :);
model1.dataTest = dataModel1(splitn{2},:);

    %% confusion chart for model1

    if ~isempty(model1.Results.mdl)
        result = basePipelinePredict( model1.Results.mdl, dataModel1(splitn{2},:), ...
            model1.Results.pipelinesweep{:} );
        accuracy = sum(result.("Response") == result.Prediction) / numel(result.("Response"));

        f1=figure();
        viz.confusionchart( result, "Response", "Prediction", ...
            "Title", "Test Data Accuracy = " + 100*accuracy + "%", ...
            "Normalization", "row-normalized", ...
            "RowSummary","row-normalized","ColumnSummary","column-normalized");
        f1.Position = [10,10,1000,700];
    else
        warning("Model1 not trained. Check input data.");
    end
else
    model1 = [];
    warning("Input data to model1 is empty or consists of insufficient classes.")
end
%% Train Stone classifiers

dataModel2 = data;
dataModel2.Partition = repmat("Train", height(dataModel2), 1);
dataModel2.Partition(splitn{2}) = "Test";

isNonStone = contains(dataModel2.Response, ["Access Sheath", ...
    "Endoscope", "Tissue-Calyx", "Tissue-Ureter", "BEGO", "Guidewire"]);
dataModel2(isNonStone,:) = [];

splitn{1} = find(dataModel2.Partition=="Train");
splitn{2} = find(dataModel2.Partition=="Test");
dataModel2.Partition = [];

% %plot PCA
% features = dataModel2{:,predictorNames};
% features = zscore(features);
% response = data.Response;
% 
% % Extract PCA features
% [coeff,score,~,~, explained] = pca(features);
% figure, gscatter(score(:,1), score(:,2), response)
% title('PCA ');
% xlabel('Feature 1')
% ylabel('Feature 2')
% check if data is not empty and if it contains at least 2 classes
if ~isempty(dataModel2) && length(unique(dataModel2.Response))>1
    [results2, info2, session2] = ops.trainAndChooseBestModel(dataModel2, splitn);
    model2.Results = results2;
    model2.Info = info2;
    model2.Session = session2;
    model2.dataTest = dataModel2(splitn{2},:);
    model2.dataTrain = dataModel2(splitn{1},:);

    %% confusion chart for model2

    if ~isempty(model2.Results.mdl)
        result = basePipelinePredict( model2.Results.mdl, dataModel2(splitn{2},:), ...
            model2.Results.pipelinesweep{:} );
        accuracy = sum(result.("Response") == result.Prediction) / numel(result.("Response"));

        f2=figure();
        viz.confusionchart( result, "Response", "Prediction", ...
            "Title", "Test Data Accuracy = " + 100*accuracy + "%", ...
            "Normalization", "row-normalized", ...
            "RowSummary","row-normalized","ColumnSummary","column-normalized");
        f2.Position = [10,10,1000,700];
    else
        disp("Model2 not trained. Check input data.");
    end
else
    model2 = [];
    warning("Input data to model2 is empty or consists of insufficient classes.")
end
end