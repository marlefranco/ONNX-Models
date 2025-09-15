classdef fitr < baseml
    %FITR Fit regression machine learning models
    %
    % Supported regression types
    %   * Regression automated ML: auto
    %   * Regression decision trees: tree
    %   * Regression support vector machine: svm
    %   * Linear regression: linear
    %   * Kernel regression: kernel
    %   * Gaussian process regression: gp
    %   * Ensemble regression learners: ensemble
    %   * Neural net regression: net, nnet<R20b
    %   * Regression trees (bagged): treebagger
    % Sytnax is standardized across model types. 
    % 
    %   To fit a model:
    %   [mdl, info] = fitr.type( tbl ) returns a trained ml model and
    %   associated evaluation metrics in an info table.
    %
    %   [mdl, info] = fitr.type( ---, "PARAM1", value1, ... ) specifies
    %   optional parameter name/value pairs for each model type. 
    %
    %   For detailed documentation on any type see help/doc fitr.type 
    %   (e.g. fitr.linear) 
    %
    %
    %   To predict response: 2 options
    %   prediction = fitr.predict( tbl, mdl ) returns model prediction as a
    %   variable
    %
    %   [tbl, infoTest] = fitr.predictandupdate( tbl, mdl ) update table
    %   with a predictioned value ( as a prediction column) and provides an 
    %   optional second output with evaluation metrics on test partition (if 
    %   specified).
    %
    %
    %   To customize hyperparameters: 
    %   params = fitr.hyperparameters( tbl, type ) returns a array of
    %   hyperparameters for specified model, which can be cusomtized an
    %   used as an arugment to OptimizeHyperparameters. Note type is a
    %   string containing one of the supported model types excluding "auto".
    %
    
    % Copyright 2021 The MathWorks Inc.
    
    methods ( Static )

        function [mdl, info] = auto( tbl, custom, options )
            %AUTO Perform automated model selection (autoML) across
            %supported regression models with option to specify hyperparameter
            %optimization.
            %
            % mdl = fitr.auto( tbl ) fit/select optimal regression model
            % using all supported regression types (see list below) and
            % mse on cross validation as the evaluation criteria. tbl
            % is a table containing predictors and reponses. If
            % PredictorNames and ResponseName arguments are not provided,
            % the default features will be all columns except the last and  
            % the default response will be the last column.
            %
            % [mdl, info] = fitr.auto( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.auto( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "Learners"     - Types of regression models to include in
            %                 automated ML. Specify as scalar string or
            %                 list of model type. Valid members include: 
            %                 "all", "tree", "svm", "linear", "kernel", 
            %                 "gp", or "ensemble". Default value is "all".
            %               
            %   "Metric"      - Evaluation critera used for model selection. 
            %                 Specify as a scalar string. Valid options
            %                 are: "mseOnCV" (default), "mseOnResub", and 
            %                 "r2OnCV".
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
            %   parameters for (note this is pointing to fitrensemble, but
            %   "HyperparameterOptimizationOptions" section will be the
            %   same). Will be updated for 2020b release.
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            % Note: This is a MathWorks Consulting implementation that
            % iterates through supported regression models with an option to
            % perform hyperparameter optimization. The final selection is
            % based on the evaluation critera specified using the 'Metric'
            % argument by default mse on cross validation. In 2020b release
            % there will be an in product implementation.
            %
            % Supported regression models
            %   * Regression decision trees: tree
            %   * Regression support vector machine: svm
            %   * Linear Regression: linear
            %   * Kernel Regression: kernel
            %   * Gaussian Process Regression: gp
            %   * Ensemble Regression Learners: ensemble
            %
            
            arguments
                tbl table              
                custom.Metric (1,1) string ...
                    {mustBeMember(custom.Metric, ["mseOnCV", "mseOnResub" "r2OnCV"])}= "mseOnCV"     
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName (1,1) string = baseml.defaultresponse( tbl );
                custom.Include (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double {mustBeInteger( custom.KFold )} = 5
                custom.Holdout (1,1) double {mustBeFinite( custom.Holdout )}= .3

                options.Learners (1,:)
                options.OptimizeHyperparameters (1,1) string ...
                    {mustBeMember(options.OptimizeHyperparameters, ["none", "auto", "all"])} = "auto";
                options.HyperparameterOptimizationOptions = struct( "UseParallel", true );
            end
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            if isempty(options.HyperparameterOptimizationOptions)
                options.HyperparameterOptimizationOptions = struct( "UseParallel", true );
            end
            
            %Optional Name/Value
            args  = namedargs2cell( options );            

             %Train
            [mdl, Optimization] = fitrauto(...
                features, ...
                response, ...
                "PredictorNames", features.Properties.VariableNames, ...
                "ResponseName", responseName, ...
                args{:});          
            
            %Model metadata
            [hypers, criterion, ~] = Optimization.bestPoint( 'Criterion', 'min-visited-mean');
            
            if ismember("learner", hypers.Properties.VariableNames )
                learnerstring = string(hypers.learner);
            else
                learnerstring = options.Learners;
            end
            
            modelType  = strcat("Regression AutoML ", learnerstring, " (fitrauto)");
            
            %Resubstitution
            mseOnResub  = mdl.loss(features, response);
            rmseOnResub = sqrt( mseOnResub );
            
            if contains(class( mdl ), 'Compact')
                predictions = predict( mdl, features );
            else
                predictions = mdl.resubPredict();
            end
            
            r2OnResub = coefficientOfDetermination( response, predictions );
                       
            if custom.CrossValidation ~= "off"
                
                mseOnCV   = exp( criterion ) -1 ;
                rmseOnCV  = sqrt( mseOnCV );
                r2OnCV        = NaN;
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
                        
        end %fitr.auto
        
        
        function [mdl, info] = autointernal( tbl, custom, options )
            %AUTO Perform automated model selection (autoML) across
            %supported regression models with option to specify hyperparameter
            %optimization.
            %
            % mdl = fitr.auto( tbl ) fit/select optimal regression model
            % using all supported regression types (see list below) and
            % mse on cross validation as the evaluation criteria. tbl
            % is a table containing predictors and reponses. If
            % PredictorNames and ResponseName arguments are not provided,
            % the default features will be all columns except the last and  
            % the default response will be the last column.
            %
            % [mdl, info] = fitr.auto( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.auto( ..., "PARAM1", value1, ... )
            % specifies optional parameter name/value pairs: 
            %   "Learners"     - Types of regression models to include in
            %                 automated ML. Specify as scalar string or
            %                 list of model type. Valid members include: 
            %                 "all", "tree", "svm", "linear", "kernel", 
            %                 "gp", or "ensemble". Default value is "all".
            %               
            %   "Metric"      - Evaluation critera used for model selection. 
            %                 Specify as a scalar string. Valid options
            %                 are: "mseOnCV" (default), "mseOnResub", and 
            %                 "r2OnCV".
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
            %   parameters for (note this is pointing to fitrensemble, but
            %   "HyperparameterOptimizationOptions" section will be the
            %   same). Will be updated for 2020b release.
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            % Note: This is a MathWorks Consulting implementation that
            % iterates through supported regression models with an option to
            % perform hyperparameter optimization. The final selection is
            % based on the evaluation critera specified using the 'Metric'
            % argument by default mse on cross validation. In 2020b release
            % there will be an in product implementation.
            %
            % Supported regression models
            %   * Regression decision trees: tree
            %   * Regression support vector machine: svm
            %   * Linear Regression: linear
            %   * Kernel Regression: kernel
            %   * Gaussian Process Regression: gp
            %   * Ensemble Regression Learners: ensemble
            % 
            %
            
             arguments
                tbl table
                custom.Learners (1,:) string ...
                    {mustBeMember(custom.Learners,["all", "tree", "svm", "linear", "kernel"...
                    "gp" "ensemble" "nnet" "treebagger"])} = "all";
                custom.Metric (1,1) string ...
                    {mustBeMember(custom.Metric, ["mseOnCV", "mseOnResub" "r2OnCV"])}= "mseOnCV"     
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
                [mdlTREE, infoTREE] = fitr.tree( tbl, args{:} );
                infos = vertcat(infos, infoTREE);
            end
            
            if any( matches( custom.Learners, ["all", "treebagger" ] ) )
                [mdlTREEB, infoTREEB] = fitr.treebagger( tbl, args{:} );
                infos = vertcat(infos, infoTREEB);
            end
            
            if any( matches( custom.Learners, ["all", "svm" ] ) )
                [mdlSVM, infoSVM] = fitr.svm( tbl, args{:} );
                infos = vertcat(infos, infoSVM);
            end
            
            if any( matches( custom.Learners, ["all", "linear" ] ) )
                [mdlLHD, infoLDH] = fitr.linear( tbl, args{:} );
                infos = vertcat(infos, infoLDH);
            end
            
            if any( matches( custom.Learners, ["all", "kernel" ] ) )
                [mdlKRN, infoKRN] = fitr.kernel( tbl, args{:} );
                infos = vertcat(infos, infoKRN);
            end
            
            if any( matches( custom.Learners, ["all", "gp" ] ) )
                [mdlGPR, infoGPR] = fitr.gp( tbl, args{:} );
                infos = vertcat(infos, infoGPR);
            end
            
            if any( matches( custom.Learners, ["all", "ensemble" ] ) )
                [mdlENS, infoENS] = fitr.ensemble( tbl, args{:} );
                infos = vertcat(infos, infoENS);
            end
            
            %Version flag
            ver = str2double(extractBetween(string(version),"R", ("a"|"b")));
            
            if any( matches( custom.Learners, ["all", "nnet" ] ) )
                if  ver >= 2021
                    [mdlNNet, infoNNet] = fitr.net( tbl, args{:} );
                else
                    [mdlNNet, infoNNet] = fitr.nnet( tbl, args{:} );
                end
                infos = vertcat(infos, infoNNet);
            end
            
            value = sortrows( infos, custom.Metric, "ascend" );
            
            switch extractBetween( value.modelType(1),"(",")" )
                case "fitrtree"
                    mdl = mdlTREE; info = infoTREE;
                case "fitrsvm"
                    mdl = mdlSVM; info = infoSVM;
                case "fitrlinear"
                    mdl = mdlLHD; info = infoLDH;
                case "fitrkernel"
                    mdl = mdlKRN; info = infoKRN;
                case "fitrgp"
                    mdl = mdlGPR; info = infoGPR;
                case "fitrensemble"
                    mdl = mdlENS; info = infoENS;
                case {"fitnet", "fitrnet"}
                    mdl = mdlNNet; info = infoNNet;
                case "TreeBagger"
                    mdl = mdlTREEB; info = infoTREEB;    
            end %switch case

            
        end %fitr.autointernal
        
        
        function [mdl, info] = tree( tbl, custom, options )
            %TREE Regression decision tree
            %
            % mdl = fitr.tree( tbl ) fit regression tree to 
            % data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.tree( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.tree( ..., "PARAM1", value1, ... )
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
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrtreeHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
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
                
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitrtree(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Regression Tree (fitrtree)";
            
            %Resubstitution
            mseOnResub      = mdl.resubLoss();
            rmseOnResub     = sqrt( mseOnResub );  
            r2OnResub       = coefficientOfDetermination( response, mdl.resubPredict() );
            
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
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(features,1),1) );
    
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.tree
        
        
        function [mdl, info] = svm( tbl, custom, options )
            %SVM Regression support vector machine
            %
            % mdl = fitr.svm( tbl ) fit regression support vector
            % machine to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.svm( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.svm( ..., "PARAM1", value1, ... )
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
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrsvmHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>  
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
                
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitrsvm(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Regression SVM (fitrsvm)";
            
            %Resubstitution
            mseOnResub      = mdl.resubLoss();
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, mdl.resubPredict() );
            
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
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(features,1),1) );
                
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.svm
        
        
        function [mdl, info] = gp( tbl, custom, options )
            %GP Gaussian Process Regression
            %
            % mdl = fitr.gp( tbl ) fit gaussian process regression
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.gp( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.gp( ..., "PARAM1", value1, ... )
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
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrgpHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>thWorks Consulting 2020   
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
                
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitrgp(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Regression Gaussian Process (fitrgp)";
            
            %Resubstitution
            mseOnResub      = mdl.resubLoss();
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, mdl.resubPredict() ); 
            
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
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(features,1),1) );
                
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
        end %fitr.gp
        
        
        function [mdl, info] = ensemble( tbl, custom, options )
            %ENSEMBLE Fit ensemble of regression learners
            %
            % mdl = fitr.ensemble( tbl ) fit ensmemble regression
            % learners to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.ensemble( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.ensemble( ..., "PARAM1", value1, ... )
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
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
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
                
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Train
            mdl = fitrensemble(...
                features, ...
                response, ...
                "ResponseName", responseName, ...
                args{:});
            
            %Model metadata
            modelType  = "Regression Ensemble Learners (fitrensemble)";
            
            %Resubstitution
            mseOnResub      = mdl.resubLoss();
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, mdl.resubPredict() );
            
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
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(features,1),1) );
                
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.ensemble
        
        
        function [mdl, info] = kernel( tbl, custom, options )
            %KERNEL fit kernel regression model
            %
            % mdl = fitr.kernel( tbl ) fit kernel regression
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.kernel( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.kernel( ..., "PARAM1", value1, ... )
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
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrkernelHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
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
                
                options.OptimizeHyperparameters = "none";
                options.HyperparameterOptimizationOptions = struct();
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train
            if options.OptimizeHyperparameters == "none"
                mdl  = fitrkernel(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    args{:});
            else
                [mdl, ~, hyperP]  = fitrkernel(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    args{:});
                
                %Append or update Hyperparameters
                hypers = table2struct( hyperP.XAtMinObjective );
                
                hyperparams = string( fieldnames( hypers ) );
                for iHyper = hyperparams(:)'
                    
                    if iscategorical( hypers.( iHyper ) )
                        hypers.( iHyper ) = string( hypers.( iHyper ) );
                    end %Corner case [ learner ]
                    
                    if isnumeric( hypers.( iHyper ) )&& isnan( hypers.( iHyper ) )
                        hypers.( iHyper ) = [];
                    end %Corner case [ epsilon ]
                    
                    options.( iHyper ) = hypers.( iHyper );
                end
                options.OptimizeHyperparameters = "none";
                args =  namedargs2cell(  options );
                
            end
            
            %Model metadata
            modelType  = "Regression Gaussian Kernel (fitrkernel)";
            
            %Resubstitution
            resub           = mdl.predict( featuresencoded.Variables );
            mseOnResub      = nansum( ( resub - response ).^2) / numel( response );
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, resub );
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), cv = cvpartition( height( featuresencoded ), "KFold", 5 );
                    case "Leaveout"
                        rng(0), cv = cvpartition( height( featuresencoded ), "Leaveout" );
                    case "Holdout"
                        rng(0), cv = cvpartition( height( featuresencoded ), "Holdout", custom.Holdout );
                end
                
                argsCV = [ args {"CVPartition", cv}];
                
                rng(0)
                mdlCV = fitrkernel(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    argsCV{:});
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(featuresencoded,1),1)./size(featuresencoded,1)...
                    );
                
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.kernel
        
        
        function [mdl, info] = linear( tbl, custom, options )
            %LINEAR Fit linear regression to high dimensional data.
            %
            % mdl = fitr.linear( tbl ) fit linear regression
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.linear( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.linear( ..., "PARAM1", value1, ... )
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
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   %TODO 
            %
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrlinearHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a> 
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
                
                options.Learner = "leastsquares"
                options.Lambda = "auto"
                options.Regularization = "ridge"
                options.OptimizeHyperparameters = "none"
                options.HyperparameterOptimizationOptions struct
                options.Weights = ones( height(tbl), 1) ./ height(tbl);
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Custom Weights
            if isa( options.Weights, 'function_handle' )
                value = options.Weights( response );
                options.Weights = value;
            elseif ~isempty( options.Weights )
                options.Weights = options.Weights( custom.Include );
            else
                options.Weights = ones( height(features), 1) ./ height(features);
            end
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train
            if options.OptimizeHyperparameters == "none"
                mdl  = fitrlinear( ...
                    featuresencoded.Variables , ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames, ...
                    args{:});
            else
                
                [mdl, ~, hyperP]  = fitrlinear(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    args{:});
                
                %Append or update Hyperparameters
                hypers = table2struct( hyperP.XAtMinObjective );
                
                hyperparams = string( fieldnames( hypers ) );
                for iHyper = hyperparams(:)'
                    
                    if iscategorical( hypers.( iHyper ) )
                        hypers.( iHyper ) = string( hypers.( iHyper ) );
                    end %Corner case [ learner ]
                    
                    if isnumeric( hypers.( iHyper ) )&& isnan( hypers.( iHyper ) )
                        hypers.( iHyper ) = [];
                    end %Corner case [ epsilon ]
                    
                    options.( iHyper ) = hypers.( iHyper );
                end
                options.OptimizeHyperparameters = "none";
                args =  namedargs2cell(  options );
                
            end
            
            %Model metadata
            modelType  = "Linear Regression High Dim (fitrlinear)";
            
            %Resubstitution
            resub           = mdl.predict( featuresencoded.Variables );
            mseOnResub      = nansum( ( resub - response ).^2) / numel( response );
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, resub );

            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                switch custom.CrossValidation
                    case "KFold"
                        rng(0), cv = cvpartition( height( featuresencoded ), "KFold", 5 );
                    case "Leaveout"
                        rng(0), cv = cvpartition( height( featuresencoded ), "Leaveout" );
                    case "Holdout"
                        rng(0), cv = cvpartition( height( featuresencoded ), "Holdout", custom.Holdout );
                end
                
                argsCV = [ args {"CVPartition", cv}];
                
                mdlCV = fitrlinear(...
                    featuresencoded.Variables, ...
                    response, ...
                    "ResponseName", responseName, ...
                    "PredictorNames", featuresencoded.Properties.VariableNames,...
                    argsCV{:});
                
                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );
                
                %predictionsOnCV=kfoldPredict( mdlCV );
                
                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    options.Weights ...
                    );
                
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.linear
        
        
        function [mdl, info] = lm( tbl, modelspec, custom, options )
            %lm Fit linear regression model by fitting to data.
            %
            % mdl = fitr.lm( tbl ) fit linear regression
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.lm( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.lm( ..., "PARAM1", value1, ... )
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
            %                 - Cross validation flag. Either 'Off', 'KFold' 
            %                 'Leaveout', 'Holdout'. Default is KFold.
            %
            %   "KFold"       - Number of folds to use if CrossValidation
            %                 is specified as 'KFold'. Default is 5.
            %
            %   TODO Add remaining NV 
            % 
            %
            
            arguments
                tbl table
                modelspec = "linear"
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Include        (:,1) logical = true( height(tbl), 1);
                custom.CrossValidation {mustBeMember(custom.CrossValidation,["off","KFold", "Leaveout", "Holdout"])} =  "KFold"
                custom.KFold (1,1) double = 5
                custom.OptimizeHyperparameters (1,1) string =  "none"
                custom.Weights        = [];
                custom.Seed (1,1) double = 0
                
                options.RobustOpts  = "on"
            end %arguments
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl( custom.Include, responseName );
            trainData       = [features response];
            
            %Custom Weights
            if isa( custom.Weights, 'function_handle' )
                value = custom.Weights( trainData.( responseName ) );
                custom.Weights= value;
            elseif ~isempty( custom.Weights)
                custom.Weights = custom.Weights( custom.Include );
            else
                custom.Weights = ones( height(features), 1);
            end
            
            warning off
            if nnz( custom.Include ) < 100
                options.RobustOpts = "off";
            end
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Linear regression
            mdl = fitlm( ...
                trainData, ...
                modelspec, ...
                'Weights', custom.Weights, ...
                args{:} );
            
            fmdl = @(x,y)fitlm( x, ...
                modelspec,...
                'Weights', y, ...
                args{:} );
            
            warning on
            
            %Model metadata
            modelType  = "Linear Regression (fitlm)";
            
            %Resubstitution
            resub           = mdl.predict( features );
            mseOnResub      = nansum( ( resub - trainData.( responseName ) ).^2) / numel( trainData.( responseName ) );
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = mdl.Rsquared.Ordinary;


            %CrossValidation
            if custom.CrossValidation ~= "off"
                
                [mseOnCV, rmseOnCV, r2OnCV] = fitr.crossval(fmdl, trainData, custom.Weights,  custom.KFold );
 
            else
                
               r2OnCV        = NaN;
               mseOnCV       = NaN;
               rmseOnCV      = sqrt( mseOnCV );
                
            end %if custom.CrossValidation
            
   
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
   
        end %fitr.lm
        
        
        function [mdl, info] = nnet( tbl, custom, options )
            %LINEAR Fit shallow neural network regression.
            %
            % mdl = fitr.nnet( tbl ) fit linear regression
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.linear( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.linear( ..., "PARAM1", value1, ... )
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
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','auto', 
            %                 'all', a string/cell array of eligible hyperparameter names, 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %    
            %   %TODO 
            %
            %   Refer to the MATLAB documentation for information on
            %   parameters for 
            %       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitrlinearHyperparameterOptimizationOptions')">Hyperparameter Optimization Options</a>
            %
            
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
                    "crossentropy","msesparse"])} = "mse"                
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
            %Convert Categorical Data
            features        = baseml.dummyvar( features );
            features        = features.Variables';
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include )';
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            %Train parameters
            if options.OptimizeHyperparameters == "none"
                net = fitnet(custom.HiddenUnit, custom.TrainFcn);
                net.performFcn = custom.PerformFcn;
                
                net.performParam.regularization = custom.Regularization;
                net.performParam.normalization = custom.Normalization;
                
                net.trainParam.showWindow = custom.ShowProgressWindow;
                
                net.divideParam.valRatio = custom.Holdout;
                net.divideParam.trainRatio = 1 - custom.Holdout;
                net.divideParam.testRatio = 0;
                
                if options.useParallel == "no"
                    mdl = train(net, features, response);
                else
                    mdl = train(net, features, response, args{1:4});
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
                fun = @(x) nnBayesOpt(features, response, ...
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
                
                net = fitnet(hypers.hu,  string(hypers.tfcn));
                net.performFcn = custom.PerformFcn;
                
                net.performParam.regularization = hypers.reg;
                net.performParam.normalization = string(hypers.norm);
                
                net.trainParam.showWindow = custom.ShowProgressWindow;
                
                net.divideParam.valRatio = custom.Holdout;
                net.divideParam.trainRatio = 1 - custom.Holdout;
                net.divideParam.testRatio = 0;
                
                % train network
                if options.useParallel == "no"
                    mdl = train(net, features, response);
                else
                    mdl = train(net, features, response, args{1:4});
                end

            end
            
            %Model metadata
            modelType  = "Regression Neural Network (fitnet)";
            
            %Resubstitution
            resub           = mdl(features);
            mseOnResub      = nansum( ( resub - response ).^2) / numel( response );
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, resub );
            
            %CrossValidation
            if custom.CrossValidation ~= "off"
                nnfitFcn = @(Xtrain, ytrain, Xtest) nnf(Xtrain, ytrain, Xtest, net);             
                mseOnCV = crossval('mse',features', response', 'Predfun',nnfitFcn, ...
                    'KFold', custom.KFold);
                rmseOnCV = sqrt( mseOnCV );
                
                nnR2Fcn = @(Xtrain, ytrain, Xtest, ytest) nnR2(Xtrain, ytrain, Xtest, ytest, net);
                r2OnCV = mean(crossval(nnR2Fcn, features', response', 'KFold', custom.KFold));
                
            else
                
               r2OnCV        = NaN;
               mseOnCV       = NaN;
               rmseOnCV      = sqrt( mseOnCV );
                
            end %if custom.CrossValidation
            
            % Create custom NNet model object
            mdl = regressionNeuralNetwork(mdl, custom.PredictorNames, ...
                custom.ResponseName);
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.nnet
        
        
        function [ mdl, info ] = net( tbl, custom )
            %NET Fit regression neural network.
            %
            % mdl = fitr.net( tbl ) regression neural network
            % to data in the table tbl. If PredictorNames and ResponseName 
            % arguments are not provided, the default features will be 
            % all columns except the last, and the default response 
            % will be the last column.
            %
            % [mdl, info] = fitr.net( tbl ) will return evaluation metrics
            % on the resubsitution, cross-validation and test sets if test
            % set is specified.
            %
            % [mdl, info] = fitr.net( ..., "PARAM1", value1, ... )
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
            %   "OptimizeHyperparameters"   
            %                 - Hyperparameters to optimize. Either 'none','all', 
            %                 or a vector of optimizableVariable objects, such as that returned 
            %                 by the 'fitr.hyperparameters' function.
            %                   
            %   "HyperparameterOptimizationOptions"     
            %                 - Options for optimization. See doc link below.   
            %   
            %   TODO Add remaining NV 
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
                custom.OptimizeHyperparameters (1,1) ...
                    {mustBeValidHyperparameter(custom.OptimizeHyperparameters)} = "none"
                custom.HyperparameterOptimizationOptions struct = struct()
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
                      
            isOptimize = isa(custom.OptimizeHyperparameters, 'optimizableVariable') || ...
                ismember(custom.OptimizeHyperparameters, ["all" "auto"]);
            
            %Train
            if ~isOptimize
                
                mdl = fitrnet(...
                    features, ...
                    response, ...
                    "ResponseName", responseName);
                
            else
                
                if isa(custom.OptimizeHyperparameters,'optimizableVariable')
                
                    hyperparams = custom.OptimizeHyperparameters;
                    
                else
                    
   
                    %Hyperparameters. For now we only have configured to an the all case.
                    hyperparams = fitr.hyperparameters(tbl, "net", ...
                        "PredictorNames", custom.PredictorNames,...
                        "ResponseName", custom.ResponseName);
                    
                    
                end %isa 

                % Check HyperparameterOptimizationOptions structure
                custom.HyperparameterOptimizationOptions = ...
                    baseml.checkHyperparamterOptimizationOptions( custom.HyperparameterOptimizationOptions );
                
                %Hyperparameter optimization options
                opts = namedargs2cell( custom.HyperparameterOptimizationOptions );
                
                results = bayesopt(...
                    @(params)i_netError(params, features, response, custom),...
                    hyperparams,...
                    opts{:});

                params = results.bestPoint();
                
                vars = string(params.Properties.VariableNames);
                tF = ismember( vars, ["Activations" "LayerWeightsInitializer" "LayerBiasesInitializer"] );
                if any(tF)

                    for iVar = vars(tF(:)')
                        params.(iVar) = string( params{:,iVar} );
                    end

                end
                params.Standardize = params.Standardize == "true";
                
                numLayers = params.NumLayers;
                params.NumLayers = [];
                
                switch numLayers
                    case 1
                        params(:, ["Layer_2_Size" "Layer_3_Size"]) = [];
                        params = renamevars(params, "Layer_1_Size", "LayerSizes");
                    case 2
                        params.("Layer_3_Size") = [];
                        params = mergevars(params, ["Layer_1_Size" "Layer_2_Size"], ...
                            "NewVariableName","LayerSizes");
                    case 3
                        params = mergevars(params, ["Layer_1_Size" "Layer_2_Size" "Layer_3_Size"], ...
                            "NewVariableName","LayerSizes");
                    otherwise
                        error( "Unhandled layer condition." )
                end
                
                args = namedargs2cell( table2struct( params ) );

                mdl = fitrnet(...
                    features, ...
                    response, ...
                    "ResponseName", responseName, ...
                    args{:});
 
            end
            
            %Model metadata
            modelType  = "Regression Neural Net (fitrnet)";
            
            %Resubstitution
            mseOnResub      = mdl.resubLoss();
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, mdl.resubPredict() );
             
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

                mseOnCV     = kfoldLoss( mdlCV );
                rmseOnCV    = sqrt( mseOnCV );

                r2OnCV  = coefficientOfDetermination( response, ...
                    kfoldPredict( mdlCV ), ...
                    ones(size(features,1),1) );  
            else
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = sqrt( mseOnCV );
            end %if custom.CrossValidation
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
            
        end %fitr.net 
              
        
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
                custom.NumTrees (1,1) double {mustBePositive(custom.NumTrees)} = 20
                custom.OptimizeHyperparameters = "none";
                custom.HyperparameterOptimizationOptions = struct();
                
                options.MinLeafSize = 5
                options.NumPredictorsToSample = [];
                options.OOBPrediction = "on"
                options.OOBPredictorImportance = "on"
                
            end
            
            %Data
            features        = tbl( custom.Include, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName )( custom.Include );
            
            
            %Optional Name/Value
            if isempty( options.NumPredictorsToSample )
                options.NumPredictorsToSample = ceil( width(features)/3 );
            end
            
            %Train
            if custom.OptimizeHyperparameters == "none"
                
                args = namedargs2cell( options ); rng( custom.Seed )
                mdl  = TreeBagger( custom.NumTrees, features, response, args{:}, ...
                    "Method", "regression");
                
            else
                                
                % Check HyperparameterOptimizationOptions structure
                custom.HyperparameterOptimizationOptions = ...
                    baseml.checkHyperparamterOptimizationOptions( custom.HyperparameterOptimizationOptions );
                
                opts = namedargs2cell( custom.HyperparameterOptimizationOptions );
                               
                %Hyperparameters. For now we only have configured to an the all case.
                hyperparams = fitc.hyperparameters(tbl, "treebagger", ...
                    "PredictorNames", custom.PredictorNames,...
                    "ResponseName", custom.ResponseName);  
                              
                optionfields = fieldnames(options);
                options = rmfield( options, optionfields(ismember(optionfields,{hyperparams.Name})) );
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
                
                rng( custom.Seed )
                mdl  = TreeBagger( custom.NumTrees, features, response, args{:}, ...
                    "Method", "regression");
                
            end
            
            %Model metadata
            modelType  = "Regression Tree Bagged (TreeBagger)";
            
            %Resubstitution
            resub           = mdl.predict( features );           
            mseOnResub      = nansum( ( resub - response ).^2) / numel( response );
            rmseOnResub     = sqrt( mseOnResub );
            r2OnResub       = coefficientOfDetermination( response, resub );
            
            % Cross validation
            if custom.CrossValidation ~= "off"
                
                [mseOnCV, rmseOnCV, r2OnCV] = ...
                    i_treeBaggerCV( features, response, custom, args );
                
            else
                
                r2OnCV        = NaN;
                mseOnCV       = NaN;
                rmseOnCV      = NaN;
                
            end %if custom.CrossValidation
            
            
            %Metrics
            info = table( modelType, ...
                r2OnResub, r2OnCV, ...
                mseOnResub, rmseOnResub,...
                mseOnCV, rmseOnCV );
 
        end %function
        
              
        function result = hyperparameters( tbl, mdlName, custom )
            %hyperparameters Return an array of optimizable hyperparameters
            %
            %  params = hyperparameters( data, modelname ) returns params
            %  array of optimizable hyperparameter objects. tbl is a matlab
            %  table containing predictors and responses; modelname is a
            %  supported regression type from the list below:
            %
            %   % Supported regression models
            %   * tree
            %   * svm
            %   * linear
            %   * kernel
            %   * gp
            %   * ensemble
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
            %
            
            arguments
                tbl table
                mdlName (1,1) string {modelValidation( mdlName)}
                custom.PredictorNames (1,:) string = baseml.defaultfeatures( tbl );
                custom.ResponseName   (1,1) string = baseml.defaultresponse( tbl );
                custom.Learner        (1,1) = "Tree"
            end
            
            switch mdlName
                case "tree"
                    mdlName = "fitrtree";
                case "svm"
                    mdlName = "fitrsvm";
                case "linear"
                    mdlName = "fitrlinear";
                case "kernel"
                    mdlName = "fitrkernel";
                case "gp"
                    mdlName = "fitrgp";
                case "ensemble"
                    mdlName = "fitrensemble";
            end %switch case
               
            %Data
            features        = tbl( :, custom.PredictorNames );
            responseName    = custom.ResponseName;
            response        = tbl.( responseName );
            
            if mdlName == "fitrensemble"
                
                result = hyperparameters( mdlName, features, response, custom.Learner );
                
            elseif mdlName == "treebagger"
                
                %Optimization variables
                result   = optimizableVariable.empty(0,1);
                result(1)  = optimizableVariable('MinLeafSize',[1,20],'Type','integer');
                if width(features) ~= 1
                    
                    if width(features)-1 == 1
                        toSample = 2;
                    else
                        toSample = size(features,2)-1;
                    end
                    
                    result(2) = optimizableVariable('NumPredictorsToSample',[1,toSample],'Type','integer');

                end
                result(3) = optimizableVariable('NumTrees',["50" "100"],'Type','categorical');
                   
            elseif mdlName == "net"
                
                %Hyperparameters (this dev 21b approach)
                result    = optimizableVariable.empty(0,1);
                
                result(1) = optimizableVariable( "NumLayers",...
                    [1,3],"Type","integer");
                
                result(2) = optimizableVariable( "Activations",...
                    ["relu", "tanh", "sigmoid", "none"],'Type','categorical');
                
                result(3) = optimizableVariable( "Standardize", ...
                    ["true", "false"],'Type','categorical');
                
                if contains("Partition",tbl.Properties.VariableNames)
                    lambdalimits = [1e-5,1e5]./height(nnz( tbl.Partition=="Train" ));
                else
                    lambdalimits = [1e-5,1e5]./height(features);
                end
                
                result(4) = optimizableVariable( "Lambda", ...
                    lambdalimits, "Type","real", "Transform", "log");
                
                result(5) = optimizableVariable( "LayerWeightsInitializer",...
                    ["glorot" "he"], 'Type','categorical');
                
                result(6) = optimizableVariable( "LayerBiasesInitializer",...
                    ["zeros" "ones"], 'Type','categorical');
                
                result(7) = optimizableVariable("Layer_1_Size",[1,300], ...
                    'Type','integer', "Transform", "log");
                
                result(8) = optimizableVariable("Layer_2_Size",[1,300], ...
                    'Type','integer', "Transform", "log");
                
                result(9) = optimizableVariable("Layer_3_Size",[1,300], ...
                    'Type','integer', "Transform", "log");
                
            else
                result = hyperparameters( mdlName, features, response );
            end
            
        end %hyperparameters
        
        
        function [result, info] = predictandupdate( tbl, mdl, modelType, options )
            %predictandupdate Predict response for machine learning model,
            %update table with prediction, and evaluate on test partition if
            %defined.
            %
            % Syntax:
            %   [result, info] = fitr.predictandupdate( data, model )
            %
            %   [result, info] = fitr.predictandupdate( data, model, ...
            %       "ResponseName", response) specify response name for
            %       models that do not include this info in their object
            %       (e.g. treebagger)
            %
            
            arguments
                tbl table
                mdl
                modelType(1,1) string = ""
                options.ResponseName (1,1) string = ""
            end
            
            result = tbl;
            
            if ~isa(mdl, "regressionNeuralNetwork" )
                predictions = baseml.predict( tbl, mdl );
            else
                net = mdl.Model;
                
                features = tbl( :, mdl.PredictorNames );
                features = baseml.dummyvar( features );
                features = features.Variables';
                
                predictions = net(features)';
            end
            
            isTableCol = @(t, thisCol) startsWith(t.Properties.VariableNames, thisCol);
            
            varName = "Prediction";
            tF = isTableCol( tbl, varName );
            
            varName = varName + (sum(tF) + 1);
            
            result.( varName ) = predictions;
            
            args = namedargs2cell(options);
            
            if contains("Partition", tbl.Properties.VariableNames) && any(tbl.Partition == "Test")
                info = fitr.evaluate( result, varName, mdl, modelType, args{:});
            else
                info = table();
            end
            
        end %predictandupdate
               
    end %public static 
    
    methods( Static, Access = private )
        
        function info = evaluate( tbl, predictionname,  mdl, modelType, options )
                %EVALUATE Evalaute model on test partition
                %
                % 
                
                arguments
                    tbl table
                    predictionname (1,1) string
                    mdl
                    modelType(1,1) string = ""
                    options.ResponseName (1,1) string = ""
                end
                
                if options.ResponseName == ""
                    responsename = mdl.ResponseName;
                else
                    responsename = options.ResponseName;
                end
                
                featurenames = mdl.PredictorNames;
                
                %Drop
                obsToTest  = tbl.Partition == "Test";
                varsToKeep = [ responsename, predictionname  ];
                
                %Transform
                evaluationData = tbl( obsToTest, varsToKeep );
                
                response    = evaluationData.( responsename );
                predictions = evaluationData.( predictionname );
                
                r2OnTest       = coefficientOfDetermination( response, predictions );
                mseOnTest     = nansum( ( predictions - response ).^2) / numel( response );
                rmseOnTest    = sqrt( mseOnTest );
                
                if modelType == ""
                    modelType = fitr.selectmdltype( mdl ) ;
                end
                
                info = table( modelType, ...
                    r2OnTest, mseOnTest, rmseOnTest );  
                
            end %evaluate
        
            
        function result = optimizablemodelslist( )
            %optimizablemodelslist Internal method used for property
            %validation
            
            result =[ "ensemble"
                "gp"
                "kernel"
                "linear"
                "svm"
                "tree"
                "net"
                "treebagger"];
            
        end %optimizablemodelslist
        
        
        function [validationMSE, validationRMSE, r2OnCV] = crossval( fmdl, tbl, weights, kFolds )
            %CROSSVAL Internal method used for linear model cross
            %validation
            
            arguments
                fmdl
                tbl table
                weights (:,1) double
                kFolds (1,1) double
            end
            
            % Perform cross-validation
            rng(0), cvp = cvpartition(height( tbl ), 'KFold', kFolds);
            
            response = tbl{:,end};
            predictions = nan( size(response) );
            for fold = 1:kFolds
                
                concatenatedPredictorsAndResponse = tbl( cvp.training( fold ), : );
                thisWeight = weights( cvp.training(fold), : );
                
                try
                    
                    % Model
                    warning off
                    linearModel = fmdl( concatenatedPredictorsAndResponse, thisWeight );
                    warning on
                    
                    foldPredictions = linearModel.predict( tbl( cvp.test( fold ), : ) );
                    predictions( cvp.test(fold) ) = foldPredictions;
                    
                catch
                    
                end
                
            end %for fold
            
            % Compute validation RMSE
            isNotMissing    = ~isnan(predictions) & ~isnan(response);
            validationMSE   = nansum( ( predictions - response ).^2) / numel(response(isNotMissing) );
            validationRMSE  = sqrt( validationMSE );
            
            r2OnCV  = coefficientOfDetermination( response, ...
                predictions, ...
                weights );
            
        end %crossval
        
        
        function modelType = selectmdltype( mdl )
            %SELECTMDLTYPE Select model type based on class of mdl
            %
            %
            
            if contains(class(mdl), 'Tree') && ~contains(class(mdl), "Bagger")
                modelType = "Regression Tree (fitrtree)";
            elseif contains(class(mdl), 'SVM')
                modelType = "Regression SVM (fitrsvm)";
            elseif contains(class(mdl), 'GP')
                modelType = "Regression Gaussian Process (fitrgp)";
            elseif contains(class(mdl), 'Ensemble')
                modelType = "Regression Ensemble Learners (fitrensemble)";
            elseif contains(class(mdl), 'Kernel')
                modelType = "Regression Gaussian Kernel (fitrkernel)";
            elseif contains(class(mdl), 'LinearModel')
                modelType = "Linear Regression (fitlm)";
            elseif contains(class(mdl), ['Regression','Linear'])
                modelType = "Linear Regression High Dim (fitrlinear)";
            elseif contains(class(mdl), "Bagger")
                modelType = "Regression Tree Bagged (TreeBagger)";
            elseif matches(class(mdl), 'regressionNeuralNetwork')
                modelType  = "Regression Neural Network (fitnet)";
            elseif matches(class(mdl), 'RegressionNeuralNetwork')
                modelType  = "Regression Neural Net (fitrnet)";
            else
                modelType = "";
            end
        end
        
    end %private static
      
end %classdef

% Property validation: local functions 
function modelValidation( input )

    tF = ismember( input, fitr.optimizablemodelslist );
    if tF == false
        error( "Unsupported model type/" )
    end

end %modelValidation

function mustBeInRange(arg,b)
    if any(arg(:) < b(1)) || any(arg(:) > b(2))
        error(['Value assigned to Data is not in range ',...
            num2str(b(1)),'...',num2str(b(2))])
    end
end %mustBeInRange

function pred = nnf(Xtrain, ytrain, Xtest, net)

net = train(net, Xtrain', ytrain');

% Test the Network
pred = net(Xtest')';

end

function r2 = nnR2(Xtrain, ytrain, Xtest, ytest, net)

net = train(net, Xtrain', ytrain');

% Test the Network
pred = net(Xtest')';

r2 = coefficientOfDetermination( ytest', pred );

end

function mse = nnBayesOpt(features, response, options)

arguments
    features double
    response double
    
    options.HiddenUnits(1,1) double
    options.Normalization(1,1) string
    options.TrainFcn(1,1) string
    options.Regularization(1,1) double  
    options.PerformFcn(1,1) string
    options.ShowProgressWindow logical
    options.Holdout(1,1) double
    options.KFold(1,1) double
    
end

net = fitnet(options.HiddenUnits, options.TrainFcn);
net.performFcn = options.PerformFcn;

net.performParam.regularization = options.Regularization;
net.performParam.normalization = options.Normalization;

net.trainParam.showWindow = options.ShowProgressWindow;

net.divideParam.valRatio = options.Holdout;
net.divideParam.trainRatio = 1 - options.Holdout;
net.divideParam.testRatio = 0;

nnfitFcn = @(Xtrain, ytrain, Xtest) nnf(Xtrain, ytrain, Xtest, net);
mse = crossval('mse',features', response', 'Predfun',nnfitFcn, ...
    'KFold', options.KFold);

end

function err = i_treeBaggerError( params, Xtrain, ytrain, custom, args)
    %i_treeBaggerError

    tF     = ismember( params.Properties.VariableNames, "NumTrees" );
    sweeps = namedargs2cell( table2struct(params(:,~tF)) );
    
    if any(tF)
       custom.NumTrees = double( string(params.NumTrees) ); 
    end
    
    rng( custom.Seed )
    mdl = TreeBagger( custom.NumTrees, Xtrain, ytrain, args{:}, sweeps{:}, ...
        "Method", "regression");
    
    err    = mdl.oobError('Mode', 'ensemble');
    
end %function

function [validationMSE, validationRMSE, r2OnCV] = i_treeBaggerCV( features, response, custom, args )
    %i_treeBaggerCV Internal method used for tree model cross validation

    arguments
        features 
        response
        custom
        args 
    end

    kFolds = custom.KFold; 
    
    % Perform cross-validation
    rng( custom.Seed ), cvp = cvpartition(height( features ), 'KFold', kFolds);

    predictions = nan( size(response) );
    for fold = 1:kFolds

        rng( custom.Seed )
        mdl = TreeBagger( custom.NumTrees, ...
            features( cvp.training(fold), :), ...
            response( cvp.training(fold) ) ,...
            args{:}, ...
            "Method", "regression");
   
        foldPredictions = mdl.predict( features( cvp.test( fold ), : ) );
        predictions( cvp.test(fold) ) = foldPredictions;

    end %for fold
    
    % Compute validation RMSE
    isNotMissing    = ~isnan(predictions) & ~isnan(response);
    validationMSE   = nansum( ( predictions - response ).^2) / numel(response(isNotMissing) );
    validationRMSE  = sqrt( validationMSE );

    r2OnCV  = coefficientOfDetermination( response, ...
        predictions );

end %crossval

function err = i_netError( params, Xtrain, ytrain, custom)
    %i_netError Helper function for bayesopt w/ neural net 

    vars = string(params.Properties.VariableNames);
     tF = ismember( vars, ["Activations" "LayerWeightsInitializer" "LayerBiasesInitializer"] );
    if any(tF)
        
        for iVar = vars(tF(:)')
            params.(iVar) = string( params{:,iVar});
        end
        
    end
    params.Standardize = params.Standardize == "true";
    
    numLayers = params.NumLayers;
    params.NumLayers = [];

    switch numLayers
        case 1
            params(:, ["Layer_2_Size" "Layer_3_Size"]) = [];
            params = renamevars(params, "Layer_1_Size", "LayerSizes");
        case 2
            params.("Layer_3_Size") = [];
            params = mergevars(params, ["Layer_1_Size" "Layer_2_Size"], ...
                "NewVariableName","LayerSizes");
        case 3
            params = mergevars(params, ["Layer_1_Size" "Layer_2_Size" "Layer_3_Size"], ...
                "NewVariableName","LayerSizes");
        otherwise
            error( "Unhandled layer condition." )
    end

    args = namedargs2cell( table2struct(params) );

    %Train
    mdl = fitrnet(...
        Xtrain, ...
        ytrain, ...
        "ResponseName", custom.ResponseName, ...
        args{:});
    
    %Only support KFold for now 
    rng(0), mdlCV = crossval( mdl, "KFold", custom.KFold );
      
    %Criterion 
    err = kfoldLoss( mdlCV );

end %function

