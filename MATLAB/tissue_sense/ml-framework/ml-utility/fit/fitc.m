classdef fitc < baseml
    %FITC Fit classification machine learning models
    %
    % multi-class classification models
    %   *classification tree: "tree"
    %   *classification discriminant analysis: "discr"
    %   *classification naive bayes: "nb" 
    %   *classification k nearest: "knn"
    %   *classification ensemble: "ensemble" 
    %   *classification multi-class svm: "ecoc" 
    %   *classification neural net: "nnet"
    %
    % two-class classification models (n==2)
    %   *classification svm:  "svm"           
    %   *classification linear: "linear"
    %   *classification kernel: "kernel"
    % Sytnax is standardized across model types.
    %
    %   To fit a model:
    %   [mdl, info] = fitc.type( tbl ) returns a trained ml model and
    %   associated evaluation metrics in an info table.
    %
    %   [mdl, info] = fitc.type( ---, "PARAM1", value1, ... ) specifies
    %   optional parameter name/value pairs for each model type. 
    %
    %   For detailed documentation on any type see help/doc fitc.type 
    %   (e.g. fitc.linear) 
    %
    %
    %   To predict response: 2 options
    %   prediction = fitc.predict( tbl, mdl ) returns model prediction as a
    %   variable
    %
    %   [tbl, infoTest] = fitc.predictandupdate( tbl, mdl ) update table
    %   with a predictioned value ( as a prediction column) and provides an 
    %   optional second output with evaluation metrics on test partition (if 
    %   specified).
    %
    %
    %   To customize hyperparameters: 
    %   params = fitc.hyperparameters( tbl, type ) returns a array of
    %   hyperparameters for specified model, which can be cusomtized an
    %   used as an arugment to OptimizeHyperparameters. Note type is a
    %   string containing one of the supported model types excluding "auto".
    %
    %
    % Nick C. Howes, Sudheer Nuggehalli
    % MathWorks 2020
    %
    
    methods (Static)
        
        function [mdl, info] = auto( tbl, custom, options )
            %AUTO Perform automated model selection (autoML) across
            %supported classification models with option to specify hyperparameter
            %optimization.
            %
            % mdl = fitc.auto( tbl ) fit/select optimal classification
            % model using bayesian optimization across classification types
            % specified by "Learners" (see list below) and . tbl is a table
            % containing predictors and reponses. If PredictorNames and
            % ResponseName arguments are not provided, the default features
            % will be all columns except the last and the default response
            % will be the last column.
            %
            % [mdl, info] = fitc.auto( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.auto( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)                 
            %
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for (note this is pointing to fitcensemble, but
            %   "HyperparameterOptimizationOptions" section will be the
            %   same). Will be updated for 2020b release.
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            % Note: This is a MathWorks Consulting implementation that
            % iterates through supported classification models with an option to
            % perform hyperparameter optimization. The final selection is
            % based on the evaluation critera specified using the 'Metric'
            % argument by default mse on cross validation. In 2020b release
            % there will be an in product implementation.
            %
              
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.Learners (1,:) %= "auto"
                options.OptimizeHyperparameters (1,1) string  {mustBeMember(options.OptimizeHyperparameters,["auto","all"])}= "auto";
                options.HyperparameterOptimizationOptions = struct( "UseParallel", true );
            end
             
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            features = featuresencoded.Variables;
            tF = any ( isnan( features ), 2 );
            
            features(tF,:) = [];
            response(tF,:) = [];
            
            %Costs 
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            [mdl, Optimization] = fitcauto(...
                features, ...
                response, ...
                "PredictorNames", featuresencoded.Properties.VariableNames, ...
                "ResponseName", responseName, ...
                args{:});
            
            
            %Model metadata
            [hypers, ~, iteration] = Optimization.bestPoint( 'Criterion', 'min-visited-mean');
            
            if ismember("learner", hypers.Properties.VariableNames )
                learnerstring = string(hypers.learner);
            else
                learnerstring = options.Learners;
            end

            modelType  = strcat("Classification AutoML ", learnerstring, " (fitcauto)");
            
            %Resubstitution
            errorOnResub = mdl.loss(features, response);
            
            %CrossValidation
            errorOnCV = Optimization.ObjectiveTrace( iteration );
            
            %Estimated 
            estimatedError = Optimization.predictObjective( hypers ); %#ok<NASGU>
            
            if contains(class( mdl ), 'Compact')
                [predictions, scores] = predict( mdl, features );
            else
                [predictions, scores] = resubPredict( mdl );

            end
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );

        end %fitc.auto
        
        function [mdl, info] = autointernal( tbl, custom, options )
            %AUTO Perform automated model selection (autoML) across
            %supported classification models with option to specify hyperparameter
            %optimization.
            %
            % mdl = fitc.auto( tbl ) fit/select optimal classification
            % model using bayesian optimization across classification types
            % specified by "Learners" (see list below) and . tbl is a table
            % containing predictors and reponses. If PredictorNames and
            % ResponseName arguments are not provided, the default features
            % will be all columns except the last and the default response
            % will be the last column.
            %
            % [mdl, info] = fitc.auto( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.auto( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)                 
            %
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for (note this is pointing to fitcensemble, but
            %   "HyperparameterOptimizationOptions" section will be the
            %   same). Will be updated for 2020b release.
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            % Note: This is a MathWorks Consulting implementation that
            % iterates through supported classification models with an option to
            % perform hyperparameter optimization. The final selection is
            % based on the evaluation critera specified using the 'Metric'
            % argument by default mse on cross validation. In 2020b release
            % there will be an in product implementation.
            %
              
           arguments
                tbl table
                custom.Learners (1,:) = "all"
                custom.Metric (1,1) string ...
                    {mustBeMember(custom.Metric, ["errorOnResub", "errorOnCV"])}= "errorOnCV"     
                options.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                options.ResponseName (1,1) string = baseml.defaultresponse( tbl );
                options.Include (:,1) logical = true( height(tbl), 1);
                options.CrossValidation {mustBeMember(options.CrossValidation,["KFold", "Leaveout", "Holdout"])} =  "KFold"
                options.KFold (1,1) double {mustBeInteger( options.KFold )} = 5
                options.Holdout (1,1) double {mustBeFinite( options.Holdout )}= .3
                options.OptimizeHyperparameters (1,1) string ...
                    {mustBeMember(options.OptimizeHyperparameters, ["none", "auto", "all"])} = "none";
                options.HyperparameterOptimizationOptions = struct();
           end
           
           %Optional Name/Value
            args = namedargs2cell( options );
            
            infos = [];
            if any( matches( custom.Learners, ["all", "tree" ] ) )
                [mdlTREE, infoTREE] = fitc.tree( tbl, args{:} );
                infos = vertcat(infos, infoTREE);
            end
            
            if any( matches( custom.Learners, ["all", "svm" ] ) )
                [mdlSVM, infoSVM] = fitc.svm( tbl, args{:} );
                infos = vertcat(infos, infoSVM);
            end
            
            if any( matches( custom.Learners, ["all", "linear" ] ) )
                [mdlLHD, infoLDH] = fitc.linear( tbl, args{:} );
                infos = vertcat(infos, infoLDH);
            end
            
            if any( matches( custom.Learners, ["all", "kernel" ] ) )
                [mdlKRN, infoKRN] = fitc.kernel( tbl, args{:} );
                infos = vertcat(infos, infoKRN);
            end
            
            if any( matches( custom.Learners, ["all", "discr" ] ) )
                [mdlDISC, infoDISCR] = fitc.discr( tbl, args{:} );
                infos = vertcat(infos, infoDISCR);
            end
            
            if any( matches( custom.Learners, ["all", "ensemble" ] ) )
                [mdlENS, infoENS] = fitc.ensemble( tbl, args{:} );
                infos = vertcat(infos, infoENS);
            end
            
            if any( matches( custom.Learners, ["all", "nnet" ] ) )
                [mdlNNet, infoNNet] = fitc.nnet( tbl, args{:} );
                infos = vertcat(infos, infoNNet);
            end
            
            if any( matches( custom.Learners, ["all", "nb" ] ) )
                [mdlNB, infoNB] = fitc.nb( tbl, args{:} );
                infos = vertcat(infos, infoNB);
            end
            
            if any( matches( custom.Learners, ["all", "knn" ] ) )
                [mdlKNN, infoKNN] = fitc.knn( tbl, args{:} );
                infos = vertcat(infos, infoKNN);
            end
            
            if any( matches( custom.Learners, ["all", "ecoc" ] ) )
                [mdlECOC, infoECOC] = fitc.ecoc( tbl, args{:} );
                infos = vertcat(infos, infoECOC);
            end
            
            value = sortrows( infos, custom.Metric, "ascend" );
            
            switch extractBetween( value.modelType(1),"(",")" )
                case "fitctree"
                    mdl = mdlTREE; info = infoTREE;
                case "fitcsvm"
                    mdl = mdlSVM; info = infoSVM;
                case "fitclinear"
                    mdl = mdlLHD; info = infoLDH;
                case "fitckernel"
                    mdl = mdlKRN; info = infoKRN;
                case "fitcnb"
                    mdl = mdlNB; info = infoNB;
                case "fitcensemble"
                    mdl = mdlENS; info = infoENS;
                case "fitcdiscr"
                    mdl = mdlDISC; info = infoDISCR;
                case "fitcknn"
                    mdl = mdlKNN; info = infoKNN;
                case "fitcecoc"
                    mdl = mdlECOC; info = infoECOC;
                case "patternnet"
                    mdl = mdlNNet; info = infoNNet;
            end %switch case

        end %fitc.autointernal
        
        
        function [mdl, info] = tree( tbl, custom, options )
            %TREE Fit a classification decision tree.
            %
            % mdl = fitc.tree( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.tree( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.tree( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitctreeHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs 
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitctree(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Tree (fitctree)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.tree
        
        
        function [mdl, info] = discr( tbl, custom, options )
            %discr Fit discriminant analysis.
            %
            % mdl = fitc.discr( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.discr( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.discr( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcdiscrHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcdiscr(...
                featuresencoded.Variables , ...
                response, ...
                "PredictorNames", featuresencoded.Properties.VariableNames, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Discriminant (fitcdiscr)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.discr
        
        
        function [mdl, info] = nb( tbl, custom, options )
            %nb Fit a Naive Bayes classifier to data.
            %
            % mdl = fitc.nb( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.nb( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.nb( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcnbHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcnb(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Naive Bayes (fitcnb)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.nb
        
        
        function [mdl, info] = knn( tbl, custom, options )
            %knn KNN classification model
            %
            % mdl = fitc.knn( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.knn( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.knn( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcknnHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcknn(...
                featuresencoded.Variables , ...
                response, ...
                "PredictorNames", featuresencoded.Properties.VariableNames, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification KNN (fitcknn)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.knn
        
        
        function [mdl, info] = svm( tbl, custom, options )
            %svm Fit a classification Support Vector Machine (SVM)
            %
            % mdl = fitc.svm( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.svm( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.svm( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcsvmHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Standardize (1,1) logical = false
                options.Verbose (1,1) double {mustBeMember(options.Verbose, [0 1 2])} = 0
                options.KernelFunction (1,1) string = "linear"
                options.KernelScale (1,1) = "auto"
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcsvm(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Support Vector Machine (fitcsvm)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
            
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.svm
        
        
        function [mdl, info] = ensemble( tbl, custom, options )
            %ensemble Fit ensemble of classification learners
            %
            % mdl = fitc.ensemble( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.ensemble( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.ensemble( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcensemble(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Ensemble Learners (fitcensemble)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
            
        end %fitc.ensemble
        
        
        function [mdl, info] = ecoc( tbl, custom, options )
            %ecoc Fit a multiclass model for Support Vector Machine or other classifiers.
            %
            % mdl = fitc.ecoc( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.ecoc( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.ecoc( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcecocHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Verbose (1,1) double {mustBeMember(options.Verbose, [0 1 2])} = 0
                options.Learners (1,1) = "svm"
                options.Options = [] 
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitcecoc(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                "FitPosterior", true, ...
                args{:});
            
            %Model metadata
            modelType  = "Classification Multi-Class Support Vector (fitcecoc)";
            
            %Resubstitution
            errorOnResub      = mdl.resubLoss();
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
                    case "Leaveout"
                        rng(0), mdlCV = crossval( mdl, "Leaveout", "on" );
                    case "Holdout"
                        rng(0), mdlCV = crossval( mdl, "Holdout", custom.Holdout );
                end
                
                errorOnCV     = kfoldLoss( mdlCV );
                
            else
                errorOnCV     = NaN;
            end %if custom.CrossValidation
            
            [predictions, scores] = resubPredict( mdl );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
      
        end %fitc.ecoc
        
        
        function [mdl, info] = kernel( tbl, custom, options )
            %kernel Fit a kernel classification model by explicit feature expansion. 
            %
            % mdl = fitc.kernel( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.kernel( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.kernel( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitckernelHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train
            if options.OptimizeHyperparameters == "none"
                mdl  = fitckernel( ...
                    featuresencoded.Variables , ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames, ...
                    args{:});
            else
                
                [mdl, fitInfo, Optimization]  = fitckernel(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    args{:});     
            end
            

            %Model metadata
            modelType  = "Classification Kernel (fitckernel)";
            
            %Resubstitution
            errorOnResub = loss(mdl, featuresencoded.Variables, response);

            %CrossValidation
            if custom.CrossValidation ~= "off"
                  
                if options.OptimizeHyperparameters == "none"
                
                    switch custom.CrossValidation
                        case "KFold"
                            rng(0), opt = {"KFold", custom.KFold};
                        case "Leaveout"
                            rng(0), opt = {"Leaveout", "on"}; %#ok<CLARRSTR>
                        case "Holdout"
                            rng(0), opt = {"Holdout" custom.Holdout};
                    end
                
                    argsCV = [ args opt];
               
                    mdlCV = fitckernel(...
                        featuresencoded.Variables, ...
                        response, ...
                        "ResponseName", responseName, ...
                        "PredictorNames", featuresencoded.Properties.VariableNames,...
                        argsCV{:});
                    
                    errorOnCV = kfoldLoss( mdlCV );
                    
                else
                    
                    %'min-observed'
                    %'min-mean'
                    %'min-upper-confidence-interval'
                    % ** 'min-visited-mean' **
                    %'min-visited-upper-confidence-interval'
                    
                    %Best point in bayesian optimization 
                    [hypers, ~, iteration] = Optimization.bestPoint( 'Criterion', 'min-visited-upper-confidence-interval');
   
                    %CrossValidation
                    errorOnCV = Optimization.ObjectiveTrace( iteration );
                    
                    %Estimated
                    estimatedError = Optimization.predictObjective( hypers );
                    
                end  
            else
                errorOnCV = NaN;
            end %if custom.CrossValidation
            
            %These linear/kernel methods don't have a resubPredict
            %[predictions, scores] = resubPredict( mdl );
            [predictions, scores] = mdl.predict( featuresencoded.Variables );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
            
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );

            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.kernel
        
        
        function [mdl, info] = linear( tbl, custom, options )
            %linear Fit a linear classification model to high-dimensional data.
            %
            % mdl = fitc.linear( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.linear( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.linear( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "PredictorNames"    
            %                 - Predictor variable names, specified as a
            %                 list of all predictors/features to be
            %                 included in the model. You must specify as a
            %                 string scalar or array. Default value is a list  
            %                 of all columns except the last in data table.
            %
            %   "ResponseName"
            %                 - Response variable name, specified as the
            %                 name of target/resposne in data table. You
            %                 must specify as a string scalar. Default 
            %                 value is last column in data table.
            %
            %   "Include"     - Logical index with length (n,1) to indicate
            %                 observations to include in training. Any zero 
            %                 elements will be test set. Default value is ones(n,1).
            %                 
            %   "CrossValidation"   
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   "Holdout"     - Fraction of the data used for holdout if
            %                 CrossVadlidation is specified as 'Holdout'. 
            %                 Default is 0.3. Must in range (0,1)
            %           
            %   "Cost"        - structure S with two fields: S.ClassificationCosts
            %                 containing the cost matrix C, and S.ClassNames
            %                 containing the class names and defining the
            %                 order of classes used for the rows and columns
            %                 of the cost matrix. Cost matrix is square
            %                 matrix, where COST(I,J) is the cost of
            %                 classifying a point into class J if its true
            %                 class is I.
            %
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitclinearHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                
                options.Cost (1,1) struct = struct()
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train
            if options.OptimizeHyperparameters == "none"
                mdl  = fitclinear( ...
                    featuresencoded.Variables , ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames, ...
                    args{:});
            else      
                [mdl, fitInfo, Optimization]  = fitclinear(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    args{:});    
            end
            
            %Model metadata
            modelType  = "Classification High Dim (fitclinear)";
            
            %Resubstitution
            errorOnResub = loss(mdl, featuresencoded.Variables, response);

            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                if options.OptimizeHyperparameters == "none"
                
                    switch custom.CrossValidation
                        case "KFold"
                            rng(0), opt = {"KFold", custom.KFold};
                        case "Leaveout"
                            rng(0), opt = {"Leaveout", "on"}; %#ok<CLARRSTR>
                        case "Holdout"
                            rng(0), opt = {"Holdout" custom.Holdout};
                    end

                    argsCV = [ args opt];

                    mdlCV = fitclinear(...
                        featuresencoded.Variables, ...
                        response, ...
                        "ResponseName", responseName, ...
                        "PredictorNames", featuresencoded.Properties.VariableNames,...
                        argsCV{:});

                    errorOnCV  = kfoldLoss( mdlCV );
                             
                else
                    
                    %'min-observed'
                    %'min-mean'
                    %'min-upper-confidence-interval'
                    % ** 'min-visited-mean' **
                    %'min-visited-upper-confidence-interval'
                   
                    %Best point in bayesian optimization 
                    [hypers, ~, iteration] = Optimization.bestPoint( 'Criterion', 'min-visited-upper-confidence-interval');
 
                    %CrossValidation
                    errorOnCV = Optimization.ObjectiveTrace( iteration );
                    
                    %Estimated
                    estimatedError = Optimization.predictObjective( hypers );
                    
                end
                
                
            else
                errorOnCV = NaN;
            end %if custom.CrossValidation
            
            %These linear/kernel methods don't have a resubPredict
            %[predictions, scores] = resubPredict( mdl );
            [predictions, scores] = mdl.predict( featuresencoded.Variables );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            classes = categories(response);
            
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );
            
            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.linear
        
        function [mdl, info] = nnet( tbl, custom, options )
            %nnet Fit a shallow neural network classification model by explicit feature expansion. 
            %
            % mdl = fitc.nnet( tbl ) fit classification tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitc.nnet( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitc.nnet( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 

            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.HiddenUnit  (1,1) {mustBeInteger(custom.HiddenUnit)} = 10;
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3          
                custom.TrainFcn (1,1) string {mustBeMember(custom.TrainFcn,["trainlm","trainbr", "trainscg", "trainrp", ...
                    "trainbfg","traincgb","traincgf","traincgp","trainoss","traingdx", "traingda"])} =  "trainscg"
                custom.PerformFcn (1,1) string {mustBeMember(custom.PerformFcn,["mae","mse","sae","sse", ...
                    "crossentropy","msesparse"])} = "crossentropy"                
                custom.Regularization (1,1) double {mustBeInRange(custom.Regularization,[0,1])} = 0
                custom.Normalization (1,1) string {mustBeMember(custom.Normalization,["none","standard", "percent"])} =  "none"
                custom.MaxEpochs (1,1) double = 1000
                custom.ShowProgressWindow (1,1) logical = false   
                custom.MaxTime (1,1) double = inf
                
                options.useParallel (1,1) string {mustBeMember(options.useParallel, ["no", "yes"])} = "no"
                options.useGPU (1,1) string {mustBeMember(options.useGPU, ["no", "yes", "only"])} = "no"
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            if ~iscategorical(response)
                response = categorical(response);
            end
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            featuresencoded = featuresencoded.Variables';
            responsesencoded = dummyvar(response)';
            
            %Optional Name/Value
            args = namedargs2cell( options );    
            
            %Train
            if options.OptimizeHyperparameters == "none"
                net = patternnet(custom.HiddenUnit, ....
                    custom.TrainFcn, custom.PerformFcn);
                
                net.performParam.regularization = custom.Regularization;
                net.performParam.normalization = custom.Normalization;
                
                net.trainParam.showWindow = custom.ShowProgressWindow;
                
                % How to separate out training and validation?
                net.divideParam.valRatio = custom.Holdout;
                net.divideParam.trainRatio = 1 - custom.Holdout;
                net.divideParam.testRatio = 0;
                
                if options.useParallel == "no"
                    mdl = train(net, featuresencoded, responsesencoded);
                else
                    mdl = train(net, featuresencoded, responsesencoded, args{1:4});
                end
            else                              
                rng(0)
                % Set optimization variables
                hu = optimizableVariable('hu',[10, 100],'Type','integer');
                norm = optimizableVariable('norm',{'none','standard','percent'},'Type','categorical');
                tfcn = optimizableVariable('tfcn',{'trainlm','trainbr', 'trainscg','trainrp', 'traingda', ...
                    'trainbfg', 'trainoss', 'traincgb', 'traingdx'},'Type','categorical');
                reg = optimizableVariable('reg',[0, 1],'Type','integer');
                
                % create loss evaluation function
                fun = @(x) nnBayesOpt(featuresencoded, response, ...
                    "HiddenUnits", x.hu, "Normalization", x.norm, ...
                    "TrainFcn", x.tfcn, "Regularization", x.reg, ...
                   "PerformFcn", custom.PerformFcn, ...
                    "ShowProgressWindow", custom.ShowProgressWindow, ...
                    "Holdout", custom.Holdout, "KFold", custom.KFold);           
                
                % hyperparameter optimization using bayesopt
                warning('off','nnet:train:NonSqrErrorFixed')
           
                % Check HyperparameterOptimizationOptions structure
                options.HyperparameterOptimizationOptions = ...
                    baseml.checkHyperparamterOptimizationOptions( options.HyperparameterOptimizationOptions );
                
                if options.HyperparameterOptimizationOptions.ShowPlots == false
                    results = bayesopt(fun,[hu, norm, tfcn, reg],'Verbose',1, ...
                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                        'PlotFcn', [], ...
                        'MaxObjectiveEvaluations',options.HyperparameterOptimizationOptions.MaxObjectiveEvaluations, ...
                        'UseParallel', options.HyperparameterOptimizationOptions.UseParallel);
                else
                    results = bayesopt(fun,[hu, norm, tfcn, reg],'Verbose',1, ...
                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}, ...
                        'MaxObjectiveEvaluations',options.HyperparameterOptimizationOptions.MaxObjectiveEvaluations, ...
                        'UseParallel', options.HyperparameterOptimizationOptions.UseParallel);
                end
                warning('on')
                
                % extract optimal hyperparameters and set up network
                hypers = bestPoint( results, 'Criterion', 'min-visited-upper-confidence-interval');
                
                net = patternnet(hypers.hu, ....
                    string(hypers.tfcn), string(custom.PerformFcn));
                
                net.performParam.regularization = hypers.reg;
                net.performParam.normalization = string(hypers.norm);
                
                net.trainParam.showWindow = custom.ShowProgressWindow;
                
                net.divideParam.valRatio = custom.Holdout;
                net.divideParam.trainRatio = 1 - custom.Holdout;
                net.divideParam.testRatio = 0;
                
                % train network
                if options.useParallel == "no"
                    mdl = train(net, featuresencoded, responsesencoded);
                else
                    mdl = train(net, featuresencoded, responsesencoded, args{1:4});
                end
              
            end

            %Model metadata
            modelType  = "Classification Neural Network (patternnet)";
            
            %Resubstitution
            scores = mdl(featuresencoded);
            predictions = scores == max(scores);            
            error = any(predictions - responsesencoded);
            errorOnResub = sum(error) / length(error);
            
            %CrossValidation
            nnfitFcn = @(Xtrain, ytrain, Xtest) nnf(Xtrain, ytrain, Xtest, net);
            
            if custom.CrossValidation ~= "off"
                errorOnCV = crossval('mcr',featuresencoded', response, 'Predfun',nnfitFcn, ...
                    'KFold', custom.KFold);                
            else
                errorOnCV = NaN;
            end %if custom.CrossValidation
            
            %F1Score                       
            classes = categories(response);
            predictions = baseml.dummyvar2cats( predictions, classes );
                        
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores', classes );
            
            % Create custom NNet model object
            mdl = ClassificationNeuralNetwork(mdl, custom.PredictorNames, ...
                custom.ResponseName, classes);
            
            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %fitc.nnet
        
        function [mdl, info] = treebagger(tbl, custom, options)
            %treebagger Bootstrap aggregation for an ensemble of decision trees.
            %
            %
            
            arguments
                tbl table
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.Holdout (1,1) double = .3
                custom.Seed (1,1) double = 0
                custom.NumTrees (1,1) double {mustBePositive(custom.NumTrees)} = 100
                custom.OptimizeHyperparameters = "none";
                custom.HyperparameterOptimizationOptions = struct(); 
                
                options.Cost (1,1) struct = struct()
                options.MinLeafSize = 1
                options.NumPredictorsToSample = [];
                options.OOBPrediction = "on" 
                options.OOBPredictorImportance = "on" 
        
            end
                   
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Costs
            if isempty(fieldnames( options.Cost ) )
                options = rmfield( options,"Cost" );
            end
            
            %Optional Name/Value
            if isempty( options.NumPredictorsToSample )
                options.NumPredictorsToSample = ceil( sqrt(width(features)) );
            end
            
            %Train
            if custom.OptimizeHyperparameters == "none"
                
                args = namedargs2cell( options );
                mdl  = TreeBagger( custom.NumTrees, features, response, args{:} );
                
            else
                
                
                if isempty( fieldnames(custom.HyperparameterOptimizationOptions) )
                    custom.HyperparameterOptimizationOptions.AcquisitionFunctionName = 'expected-improvement-plus';
                    custom.HyperparameterOptimizationOptions.MaxObjectiveEvaluations = 2;
                    custom.HyperparameterOptimizationOptions.Verbose = 1;
                    
                    opts = namedargs2cell( custom.HyperparameterOptimizationOptions );
                    
                end
                
                %Optimization variables
                list = "MinLeafSize";
                hyperparams   = optimizableVariable.empty(0,1);
                hyperparams(1)  = optimizableVariable('MinLeafSize',[1,20],'Type','integer');
                if width(features) ~= 1
                    
                    if width(features)-1 == 1
                        toSample = 2;
                    else
                        toSample = size(features,2)-1;
                    end
                    
                    hyperparams(2) = optimizableVariable('NumPredictorsToSample',[1,toSample],'Type','integer');
                    list(2) = "NumPredictorsToSample";
                end

                if custom.OptimizeHyperparameters == "all"
                    hyperparams(3) = optimizableVariable('NumTrees',["50" "100"],'Type','categorical');
                end
                    
                options = rmfield( options, list );
                args    = namedargs2cell( options );
                
                results = bayesopt(...
                    @(params)i_treeBaggerError(params, features, response, custom, args),...
                    hyperparams,...
                    opts{:});
                
                optimalparams = results.bestPoint();
                
                tF   = ismember( optimalparams.Properties.VariableNames, "NumTrees" ); 
                args = [args namedargs2cell( table2struct( optimalparams(:,~tF) ) )];
  
                if any( tF )
                    custom.NumTrees = double( string(optimalparams.NumTrees) );
                end
                
                mdl  = TreeBagger( custom.NumTrees, features, response, args{:} );
           
            end 
            
            %Model metadata
            modelType  = "Classification Neural Network (patternnet)";
            
            %Resubstitution
            [predictions, scores] = mdl.predict( features );
            
            if iscellstr( predictions ) %#ok<ISCLSTR>
                response = categorical( response );
                predictions = categorical( predictions );   
            end    
            errorOnResub = nnz(predictions ~= response)/numel( response  );  
            
            % Cross validation 
            if custom.CrossValidation ~= "off"
                errorOnCV = crossval('mcr', features, response, ...
                    'Predfun',@(Xtrain, ytrain, Xtest)i_treeBagger(Xtrain, ytrain, Xtest, custom, args), ...
                    'KFold', custom.KFold);
            else
                errorOnCV = NaN;
            end %if custom.CrossValidation
            
            %Compute Precision, Recall, F1 Score, and AUC
            classes = categories(response);
            [precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain] = ...
                computeMetrics( response, predictions, scores, classes );
            
            %Metrics
            info = table( modelType, errorOnResub, errorOnCV, ...
                precisionOnTrain, recallOnTrain, f1ScoreOnTrain, AUCOnTrain );
            
        end %function
        
        function [result, info] = predictandupdate( tbl, mdl, modelType )
            %predictandupdate Predict response for machine learning model,
            %update table with prediction, and evaluate on test partition if
            %defined.
            %
            % TODO Syntax
            %
            %   N.C.Howes, Sudheer Nuggehalli
            %   MathWorks 2020
            
            arguments
                tbl table
                mdl
                modelType(1,1) string = ""
                
            end
            
            result = tbl;
            
            if ~isa(mdl, "NeuralNetwork" )
                predictions = baseml.predict( tbl, mdl );
            else
                net = mdl.Model;
                
                try
                    features = tbl( :, mdl.PredictorNames );
                    features = features.Variables';
                catch
                    featuresencoded = baseml.dummyvar( tbl );
                    features =  featuresencoded( :, mdl.PredictorNames );
                    features = features.Variables';
                end
                
                response = tbl.( tbl.Properties.CustomProperties.Response );
                
                if ~iscategorical(response)
                    response = categorical(response);
                end
                
                scores = net( features );
                predictions = scores == max(scores);
                
                classes = string(categories(response));
                predictions = baseml.dummyvar2cats( predictions, classes );
               
            end
           
            isTableCol = @(t, thisCol) startsWith(t.Properties.VariableNames, thisCol);
            
            varName = "Prediction";
            tF = isTableCol( tbl, varName );
            
            varName = varName + (sum(tF) + 1);
            
            result.( varName ) = predictions;
            
            if contains("Partition", tbl.Properties.VariableNames) && any(tbl.Partition == "Test")
                info = fitc.evaluate( result, varName, mdl, modelType );           
            else
                info = table();
            end
            
        end %predictandupdate

        
        function info = evaluate( tbl, predictionname,  mdl, modelType )
            %EVALUATE Evalaute model on test partition
            %
            %
            
            arguments
                tbl table
                predictionname (1,1) string
                mdl
                modelType(1,1) string = ""
            end
            
            responsename = mdl.ResponseName;
            featurenames = mdl.PredictorNames;
            
            %Drop
            obsToTest  = tbl.Partition == "Test";
            varsToKeep = [ responsename, predictionname  ];
            
            %Transform
            evaluationData = tbl( obsToTest, varsToKeep );
            
            response    = evaluationData.( responsename );
            predictions = evaluationData.( predictionname );
            
            if modelType == ""
                modelType = fitc.selectmdltype( mdl ) ;
            end
            
            if ~(iscategorical(response) && iscategorical(predictions))
                    response = categorical(response);
                    predictions = categorical(predictions);
            end
            
            errorOnTest = nansum( predictions ~= response ) / numel( response );
            
            if ~isa(mdl, "NeuralNetwork" )
                try
                    features = tbl( obsToTest, featurenames );
                catch
                    featuresencoded = baseml.dummyvar( tbl(obsToTest,:) );
                    features =  featuresencoded( :, featurenames );
                end
                
                try
                    [~,scores] = mdl.predict( features );
                catch
                    [~,scores] = mdl.predict( features.Variables );
                end
            else
                net = mdl.Model;
                
                try
                    features = tbl( obsToTest , featurenames );
                    features = features.Variables';
                catch
                    featuresencoded = baseml.dummyvar( tbl(obsToTest,:) );
                    features =  featuresencoded( :, featurenames );
                    features = features.Variables';
                end
                
                scores = net( features )';             
            end
            
            classes = categories(response);
            
            % Compute Precision, Recall, F1 Score, and AUC
            [precisionOnTest, recallOnTest, f1ScoreOnTest, AUCOnTest] = ...
                computeMetrics( response, predictions, scores, classes );
                                    
            info = table( modelType, errorOnTest, precisionOnTest, recallOnTest, ...
                f1ScoreOnTest, AUCOnTest );
       
        end %evaluate

        
        function result = hyperparameters( tbl, mdlName, custom )
            %HYPERPARAMETERS
            %hyperparameters Return an array of optimizable hyperparameters
            %
            %  params = hyperparameters( data, modelname ) returns params
            %  array of optimizable hyperparameter objects. tbl is a matlab
            %  table containing predictors and responses; modelname is a
            %  supported regression type from the list below:
            %
            %   % Supported classification models
            %   * tree
            %   * svm
            %   * linear
            %   * kernel
            %   * nb
            %   * knn
            %   * discr
            %   * ensemble
            %   * cecoc
            %
            % params = hyperparameters( data, modelname, ...
            %    "PredictorNames", listOfPredictors, ...
            %    "ResponseName", responsename) specify a list of predictor
            %    and the response varible in the table data
            %
            % params = hyperparameters( ..., "Learner", learnerType) returns 
            % array of optimizable hyperparameter objects when modelname is
            % ensemble. learnerType is 'Tree', or a template of a listed learner.
            %
            
            arguments
                tbl table
                mdlName (1,1) string {modelValidation( mdlName)}
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Learner (1,1) string = "Tree"
            end
            
            
            switch mdlName
                case "tree"
                    mdlName = "fitctree";
                case "svm"
                    mdlName = "fitcsvm";
                case "linear"
                    mdlName = "fitclinear";
                case "kernel"
                    mdlName = "fitckernel";
                case "nb"
                    mdlName = "fitcnb";
                case "knn"
                    mdlName = "fitcknn";
                case "discr"
                    mdlName = "fitcdiscr";
                case "ensemble"
                    mdlName = "fitcensemble";
                case "ecoc"
                    mdlName = "fitcecoc";
            end %switch case
            
            
            %Data
            features        = tbl( :, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName );
            
            if mdlName == "fitcecoc" || mdlName == "fitcensemble"
                result = hyperparameters( mdlName, features, response, custom.Learner );
            else
                result = hyperparameters( mdlName, features, response );
            end
            
        end %hyperparameters
        
        
        function result = optimizablemodelslist( )
            %optimizablemodelslist 
            
            result = [ "tree"
                "discr"
                "nb"
                "knn"
                "svm"
                "ensemble"
                "ecoc"
                "kernel"
                "linear"];
     
        end %optimizablemodelslist
        
           
    end %static
    
    methods (Static, Access = private)
        
        function modelType = selectmdltype( mdl )
            %SELECTMDLTYPE Select model type based on class of mdl
            %
            %
            
            if contains(class(mdl), 'Tree')
                modelType = "Classification Tree (fitctree)";
            elseif contains(class(mdl), 'SVM')
                modelType = "Classification Support Vector Machine (fitcsvm)";
            elseif contains(class(mdl), 'Discriminant')
                modelType = "Classification Discriminant (fitcdiscr)";
            elseif contains(class(mdl), 'KNN')
                modelType = "Classification KNN (fitcknn)";
            elseif contains(class(mdl), 'NaiveBayes')
                modelType = "Classification Naive Bayes (fitcnb)";
            elseif contains(class(mdl), 'Ensemble')
                modelType = "Classification Ensemble Learners (fitcensemble)";
            elseif contains(class(mdl), 'ECOC')
                modelType = "Classification Multi-Class Support Vector (fitcecoc)";
            elseif contains(class(mdl), 'Kernel')
                modelType = "Classification Kernel (fitckernel)";
            elseif contains(class(mdl), 'Linear')
                modelType = "Classification High Dim (fitclinear)";
            elseif contains(class(mdl), 'network')
                modelType  = "Classification Neural Network (patternnet)";
            else
                modelType = "";
            end
        end %selectmdltype
        
    end %private static
    
    
end %classdef

% Local functions 
function modelValidation( input )

    tF = ismember( input, fitc.optimizablemodelslist );
    if tF == false
        error( "Unsupported model type/" )
    end

end %modelValidation

function [precisionTable, recallTable, f1ScoreTable, AUCTable] = computeMetrics( response, predictions, scores, classes )

arguments
   response (:,1) categorical
   predictions (:,1) categorical
   scores double
   classes cell
end

cm = confusionchart(response, predictions, 'Visible', 'off'); set(gcf, 'Visible', 'off')
cm_mat = cm.NormalizedValues;

precision = diag(cm_mat) ./ sum(cm_mat)';
precisionTable = array2table(precision', 'VariableNames', classes);
precisionTable.Avg = mean(precision);

recall = diag(cm_mat) ./ sum(cm_mat,2);
recallTable = array2table(recall', 'VariableNames', classes);
recallTable.Avg = mean(recall);

f1Score = 2*(precision .* recall) ./ (precision + recall);
f1ScoreTable = array2table(f1Score', 'VariableNames', classes);
f1ScoreTable.Avg = mean(f1Score);

indArray = 1:length(classes);
AUC = zeros(1,length(classes));

for ii = 1:length(classes)
    idx = indArray == ii;
    diffscore =  scores(:,ii) - max(scores(:,~idx),[],2);
    [~,~,~,AUC(ii)] = perfcurve(response, diffscore, classes{ii});
end

AUCTable = array2table(AUC, 'VariableNames', classes);
AUCTable.Avg = mean(AUC);

end

% Custom validation function
function mustBeInRange(arg,b)
    if any(arg(:) < b(1)) || any(arg(:) > b(2))
        error(['Value assigned to Data is not in range ',...
            num2str(b(1)),'...',num2str(b(2))])
    end
end

function predFinal = nnf(Xtrain,ytrain,Xtest,net)

ytrainencoded = dummyvar(ytrain)';

classes = string(categories(ytrain));

net = train(net, Xtrain', ytrainencoded);

% Test the Network
yfit = net(Xtest');

predFinal = baseml.dummyvar2cats( yfit, classes );

end %function 


function error = nnBayesOpt(features, response, options)

arguments
    features double
    response categorical
    
    options.HiddenUnits(1,1) double
    options.Normalization(1,1) string
    options.TrainFcn(1,1) string
    options.Regularization(1,1) double  
    options.PerformFcn(1,1) string
    options.ShowProgressWindow logical
    options.Holdout(1,1) double
    options.KFold(1,1) double
    
end

net = patternnet(options.HiddenUnits, ....
    options.TrainFcn, options.PerformFcn);

net.performParam.regularization = options.Regularization;
net.performParam.normalization = options.Normalization;

net.trainParam.showWindow = options.ShowProgressWindow;

net.divideParam.valRatio = options.Holdout;
net.divideParam.trainRatio = 1 - options.Holdout;
net.divideParam.testRatio = 0;

nnfitFcn = @(Xtrain, ytrain, Xtest) nnf(Xtrain, ytrain, Xtest, net);

error = crossval('mcr',features', response, 'Predfun',nnfitFcn, ...
    'KFold', options.KFold);

end


function yFit = i_treeBagger(Xtrain,ytrain, Xtest, custom, args)
    %i_treeBagger

    mdl = TreeBagger( custom.NumTrees, Xtrain, ytrain, args{:} );

    yFit = mdl.predict( Xtest );

end %function 


function err = i_treeBaggerError( params, Xtrain, ytrain, custom, args)

    tF     = ismember( params.Properties.VariableNames, "NumTrees" );
    sweeps = namedargs2cell( table2struct(params(:,~tF)) );
    
    if any(tF)
       custom.NumTrees = double( string(params.NumTrees) ); 
    end
    
    mdl    = TreeBagger( custom.NumTrees, Xtrain, ytrain, args{:}, sweeps{:} );
    
    err    = mdl.oobError('Mode', 'ensemble');
    
end %function
