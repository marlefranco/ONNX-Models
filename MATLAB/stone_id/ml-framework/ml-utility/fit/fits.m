classdef fits < baseml
    %FITS Fit semi-supervised machine learning models
    %
    % fits methods:
    %
    %   Training 
    %   graph            - graph-based semi-supervised classification model       
    %   self             - self-training semi-supervised classification model
    %
    %
    %   Prediction 
    %   predict          - 
    %   predictandupdate - 
    %
    %   
    %   Evaluation
    %   evaluate         - return evaluation metrics 
    %
    
    % Copyright 2021 The MathWorks Inc.
    %
    
    methods (Static)
        function [mdl, info] = graph( labeled, unlabeled, custom, options )
            %GRAPH
            %TODO
            
            arguments
                labeled table
                unlabeled table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( labeled );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( labeled );
                custom.Include        (:,1) logical = true( height(labeled), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Method (1,1) string ...
                    {mustBeMember(options.Method,[...
                    "labelpropagation"
                    "labelspreading"
                    "labelspreadingexact"
                    "labelpropagationexact"])} = "labelpropagation"
                options.Distance = "euclidean";
            end
            
            %Data
            features        = labeled( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = labeled.( responseName )( custom.Include );
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train
            mdl = fitsemigraph(...
                features, ...
                response, ...
                unlabeled, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Graph Based Semi Supervised (fitsemigraph)";
            
            %Resubstitution
            predictions = mdl.predict( features );
            errorOnResub = nnz( predictions ~= response ) / numel( response );
            
            %Cross validation
            switch custom.CrossValidation
                case "KFold"
                    rng(0), opt = {"KFold", custom.KFold};
                case "Leaveout"
                    rng(0), opt = {"Leaveout", "on"}; %#ok<CLARRSTR>
                case "Holdout"
                    rng(0), opt = {"Holdout" custom.Holdout};
            end
            
            cvp     = cvpartition(height( features ), opt{:});
            kFolds  = cvp.NumTestSets;
            
            if isnumeric(response)
                predictions = nan( numel(response), 1 );
            else
                predictions( numel(response), 1 ) = categorical( {''}, categories(response) );
            end
            
            for fold = 1:kFolds
                
                
                mdlcv = fitsemigraph(...
                    features(cvp.training( fold ), :), ...
                    response(cvp.training( fold )), ...
                    features(cvp.test( fold ), :), ...
                    "ResponseName", responseName, ...
                    args{:});
                
                predictions( cvp.test(fold) ) = mdlcv.FittedLabels;
                
            end
            
            errorOnCV = nnz(response ~= predictions) / numel(response);
            
            %Metrics
            info = table( modelType, ...
                errorOnResub, ...
                errorOnCV );
      
        end %fits.graph
        
                   
        function [mdl, info] = self( labeled, unlabeled, custom )
            %SELF
            %TODO
            
            arguments
                labeled table
                unlabeled table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( labeled );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( labeled );
                custom.Include        (:,1) logical = true( height(labeled), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
            end
            
            %Data
            features        = labeled( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = labeled.( responseName )( custom.Include );
            
            %Optional Name/Value
            %args = namedargs2cell( options );
            
            %Train
            mdl = fitsemiself(...
                features, ...
                response, ...
                unlabeled, ...
                "ResponseName", responseName);
            
            %Model metadata
            modelType  = "Self Training Semi Supervised (fitsemiself)"; 
            
            %Resubstitution
            predictions = mdl.predict( features );
            errorOnResub = nnz( predictions ~= response ) / numel( response );
            
            %Cross validation
            switch custom.CrossValidation
                case "KFold"
                    rng(0), opt = {"KFold", custom.KFold};
                case "Leaveout"
                    rng(0), opt = {"Leaveout", "on"}; %#ok<CLARRSTR>
                case "Holdout"
                    rng(0), opt = {"Holdout" custom.Holdout};
            end
            
            cvp     = cvpartition(height( features ), opt{:});
            kFolds  = cvp.NumTestSets;
            
            if isnumeric(response)
                predictions = nan( numel(response), 1 );
            else
                predictions( numel(response), 1 ) = categorical( {''}, categories(response) );
            end
            
            for fold = 1:kFolds
                
                mdlcv = fitsemiself(...
                    features(cvp.training( fold ), :), ...
                    response(cvp.training( fold )), ...
                    features(cvp.test( fold ), :), ...
                    "ResponseName", responseName);
                
                predictions( cvp.test(fold) ) = mdlcv.FittedLabels;
                
            end %for fold 
            
            errorOnCV = nnz(response ~= predictions) / numel(response);
            
            %Metrics
            info = table( modelType, ...
                errorOnResub, ...
                errorOnCV );
             
        end %fits.self
        
        function [result, score] = predict(tbl, mdl)
            
            arguments
                tbl table
                mdl
            end
            
            try
                features = tbl( :, mdl.PredictorNames );
            catch
                featuresencoded = baseml.dummyvar( tbl );
                features = featuresencoded( :, mdl.PredictorNames );
            end %try/catch
            
            try
                [result, score] = mdl.predict( features );
            catch
                [result, score] = mdl.predict( features.Variables );
            end %try/catch 
            
            
        end %pre
        
        
        function [result, info] = predictandupdate( tbl, mdl, modelType )
            
            arguments
                tbl table
                mdl
                modelType(1,1) string = ""
            end
            
            result = tbl;
            
            [predictions, scores] = fits.predict( tbl, mdl );
            
            isTableCol = @(t, thisCol) startsWith(t.Properties.VariableNames, thisCol);
            
            varName = "Prediction";
            tF = isTableCol( tbl, varName );
            
            varName = varName + (sum(tF) + 1);
            
            result.( varName ) = predictions;
            
            if contains("Partition", tbl.Properties.VariableNames) && any(tbl.Partition == "Test")
                info = fits.evaluate( result, varName, mdl, scores, modelType );
            else
                info = table();
            end
            
        end %function
        
        
        function info = evaluate( tbl, predictionname,  mdl, scores, modelType )
            %EVALUATE Evalaute model on test partition
            %
            %
            
            arguments
                tbl table
                predictionname (1,1) string
                mdl
                scores double
                modelType(1,1) string = ""
            end
            
            responsename = tbl.Properties.CustomProperties.Response;
            
            %Drop
            obsToTest  = tbl.Partition == "Test";
            varsToKeep = [ responsename, predictionname  ];
            
            %Transform
            evaluationData = tbl( obsToTest, varsToKeep );
            
            response    = evaluationData.( responsename );
            predictions = evaluationData.( predictionname );
            scores      = scores( obsToTest, : );
            
            if modelType == ""
                modelType = fits.selectmdltype( mdl ) ;
            end
            
            errorOnTest = nansum( predictions ~= response ) / numel( response );

            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
            
            %F1 Score
            f1ScoreOnTest = computeF1Score( response, predictions, classes );
            
            %AUC of ROC Curve
            AUCOnTest = computeAUC( response, scores, classes );
                                    
            info = table( modelType, errorOnTest, ...
                f1ScoreOnTest, AUCOnTest );
       
        end %evaluate

    end %static
    
    
    methods (Static, Access = private)
        
        function modelType = selectmdltype( mdl )
            %SELECTMDLTYPE Select model type based on class of mdl
            %
            %
            
            if contains(class(mdl), 'GraphModel')
                modelType = "Graph Based Semi Supervised (fitsemigraph)";
            elseif contains(class(mdl), 'SelfTrainingModel')
                modelType = "Self Training Semi Supervised (fitsemiself)"; 
            else
                modelType = "";
            end
            
        end %function
        
    end %methods
    
end %classdef 

% Helper functions
function f1ScoreTable = computeF1Score( response, predictions, classes )

arguments
   response (:,1) categorical
   predictions (:,1) categorical
   classes cell
end

cm = confusionchart(response, predictions, 'Visible', 'off'); set(gcf, 'Visible', 'off')
cm_mat = cm.NormalizedValues;

precision = diag(cm_mat) ./ sum(cm_mat)';
recall = diag(cm_mat) ./ sum(cm_mat,2);

f1Score = 2*(precision .* recall) ./ (precision + recall);
f1ScoreTable = array2table(f1Score', 'VariableNames', classes);
f1ScoreTable.AvgScore = mean(f1Score);

end


function AUCTable = computeAUC( response, scores, classes)

arguments
    response (:,1) categorical
    scores double
    classes cell
end

indArray = 1:length(classes);
AUC = zeros(1,length(classes));

for ii = 1:length(classes)
    idx = indArray == ii;
    diffscore =  scores(:,ii) - max(scores(:,~idx),[],2);
    [~,~,~,AUC(ii)] = perfcurve(response, diffscore, classes{ii});
end

AUCTable = array2table(AUC, 'VariableNames', classes);
AUCTable.AvgAUC = mean(AUC);

end