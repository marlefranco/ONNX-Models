classdef Regression < experiment.Base & experiment.mixin.Regression
    %experiment.Regression Create a machine learning regression experiment. Each
    %run in an experiment consists of: (1) a unique data pipeline configuration,
    %(2) a selected ml model, and (3) a model/hyperparamter configuration.
    %
    % General Syntax:
    %  session = experiment.Regression(...
    %       "Data", thisData, ...
    %       "DataFcn", @(x, settings)thisPipeline(x, settings{:}), ...
    %       "Model", thisModel, ...
    %       "DataConfiguration", theseParameters, ...
    %       "ModelConfiguration", theseParameters, ...
    %       "Description, "Some description of experiment" )
    %
    % Description:
    %   experiment.Regression( "Data", dataset, ...
    %       "DataFcn", @(x, settings)datapipeline(x, settings{:} ) will create a
    %       regression experiment using the default pipeline. In a standard
    %       experiment the "Data" will be processed/prepared using the
    %       "DataFcn" specified above and automated machine learning
    %       (autoML) applied. By default, cross-validation is used (kFold =
    %       5) and hyperparameter optimization is not performed. To change
    %       these options specify "ModelConfiguration" parameters (see
    %       below).
    %       
    %       Data: is a table dataset 
    %       DataFcn: is an anonymous function specified in the format above
    %
    %   experiment.Regression( __, ...
    %       "Model", modelselection) specifies an optional model selection.
    %    
    %       Model: is a scalar string. Specify: "automl", "selectml",
    %       "stackml" or a learner from the regression model set below.
    %       Configure the model selection via the ModelConfiguration
    %       property described below. 
    %       
    %       Automated Machine Learning 
    %           *automated ML (all regression models): "automl"
    %           *select ML ('best' regression model): "selectml"
    %           *stacked ML (meta-model combining all regressions): "stackml"    
    %
    %       regression models
    %           *regression tree: "tree"
    %           *regression svm:  "svm"
    %           *regression ensemble: "ensemble" 
    %           *regression kernel: "kernel" 
    %           *regression linear: "linear" 
    %           *regression gaussian process: "gp"
    %           *regression tree bagger: "treebagger"
    %           *regression neural net: "nnet"
    %
    %   experiment.Regression( __, ...
    %       "DataConfiguration", dataparameters, ... "ModelConfiguration",
    %       modelparameters ) will create a series of experimental runs.
    %       These runs will contain unique combinations of
    %       "DataConfiguration" and "ModelConfiguration" parameters.
    %       Default search/selection method "Search" is gridsearch, which
    %       will find all unique parameter combinations using
    %       "NumGridDimensions" for each dimension (configured parameter).
    %       "DataConfiguration" contains parameters applied in the data
    %       preparation pipeline specified by "DataFcn".
    %       "ModelConfiguration" contains parameters applied in the model
    %       selection specified by "Model".
    %
    %       DataConfiguration: is an optimizeParameter array (default empty)
    %       ModelConfiguration: is an optimizeParameter array (default empty)
    %
    %       Refer to the <a href="matlab:help('optimizeParameter')">optimizeParameter</a>
    %       documentation for information on usage.
    %
    %       Valid parameters for DATACONFIGURATION include any parameter 
    %       defined in the arguments block of selected data preparation pipeline
    %       (e.g. "DataFcn"). Note that it is possible to validate
    %       DataConfiguration options prior to running the experiment. See
    %       example below. 
    %
    %       Example: Configure pipeline with/without feature selection
    %       dataConfiguration = optimizeParameter.new(...
    %       	"Name", "Selection", ...
    %           "Range", ["none" "fsrf"],...
    %           "Type", "Discrete");
    %
    %       Valid parameters for MODELCONFIGURATION for all model
    %       selection include:
    %           - OptimizeHyperparameters
    %           - HyperparameterOptimizationOptions        
    %           - CrossValidation
    %           - KFold
    %           - HoldOut        
    %           - PredictorNames
    %           - ResponseNames
    %           - Include
    %
    %       Note that there are additional options specific to automated ML.
    %
    %       For automl and stackml these options include:
    %           - Learners
    %       For selectml these options include:
    %           - Learners
    %           - Metric 
    %
    %       Example: Configure model with/without hyperparameter optimization
    %       modelConfiguration = optimizeParameter.new(...
    %       	"Name", "OptimizeHyperparameters", ...
    %           "Range", ["none" "all"],...
    %           "Type", "Discrete");
    %       
    %       Refer to <a href="matlab:help('fitr')">fitr</a> documentation for general information on
    %       parameter usage, or <a href="matlab:help('fitr/tree')">fitr/tree</a> for a specific method.      
    %       
    %   experiment.Regression( __, ...
    %       "AdditionalConfiguration", addnlparameters ... "Search",
    %       searchmethod, ... "Description", experimentdescription, ...
    %       "CustomProperties", listofcustomproperties) allows for
    %       additional customization of the experiment.
    %       "AdditionalConfiguration" allows for additional parameters to
    %       be applied in the data pipeline "DataFcn" without entering the
    %       experimental design. These parameters will be constants across
    %       all runs. "Search" specifies the parameter selection method:
    %       "randomsearch" or "gridsearch". "Description" specifies a tag
    %       or metadata associated with the experiment. "CustomProperties"
    %       specifies a set of custom properties defined in the "DataFcn"
    %       for display of the experimental results. Additionally, specify
    %       the number of samples using "Evaluation" if "Search" is
    %       "randomsearch" or number of grid samples per dimension using
    %       "NumGridDimensions" if "Search" is "gridsearch".
    %
    %   experiment.Regression methods:
    % 
    %   validate      - validate the input configruation (DataConfiguration) 
    %   build         - construct experimental runs from input configuration
    %   configure     - validate and build in a single step 
    %   preview       - preview the experimental runs as a table
    %
    %   prepare       - run the data preparation pipelines
    %   fit           - fit model to data pipeline result
    %   run           - prepare and train in a single step
    %
    %   describe      - view input parameters with model results as a table
    %   sort          - rank experiment model runs by user defined metric
    %   select        - select/extract a model from the experiment
    %   view          - view dataset corresponding to an experiment run
    %   
    %   plotmetric    - visualize mse of model runs as bar graph
    %   predictactual - visualize model runs as predict-actual scatterplot
    %
    %   save          - export experiment items/models to disk 
    %   writedata     - export trial data results with predictions
    %
    %   experiment.Regression properties
    %   
    %   Data                    - Input data
    %   DataFcn                 - Specify data preparation pipeline 
    %   Model                   - Select regression method
    %
    %   DataConfiguration       - Configure 'sweepable' parameters for data preparation pipeline
    %   ModelConfiguration      - Configure 'sweepable' parameters for model selection
    %   AdditionalConfiguration - Configure parameters for data pipeline (not sweepable in the design)
    %
    %   Search                  - Specify parameter selection method
    %   Evaluations             - Specify number of samples if "Search" is "randomsearch"   
    %   NumGridDivisions        - Specify number of grid samples if "Search" is "gridsearch"
    %
    %   Description             - Description/tag associated with the experiment  
    %   CustomProperties        - Specify properties defined in data pipeline to appear in evaluation metrics 
    %
    %
    %   N.C. Howes, Sudheer Nuggehalli
    %   Copyright 2021 The MathWorks Inc.
    %
    
    properties
        %Model Select regression method. Either automated machine learning
        %or a supported regression method from list below. Note that these
        %options are further configured using the ModelConfiguration
        %property.
        %
        % automl(default) - return multiple regression models (default all). A
        % subset of models can be specified by setting "Learners" in the
        % ModelConfiguration. 
        %
        % selectml - run multiple regression methods and return the
        % optimal model based a specified evaluation criteria (default
        % mseOnCV). The evaluation criteria can be specified by setting
        % "Metric" in the ModelConfiguration.
        %
        % stackml - return multiple regression models and a stacked
        % meta model result that combines the multiple methods into a
        % single prediction.
        %
        % tree - return regression trees 
        %
        % svm - return regression support vector machine
        %
        % ensemble - return regression ensemble
        %
        % gp - return a gaussian process 
        %
        % linear - return a linear regression
        % 
        % kernel - return a kernel regression
        %  
        % nnet - return a neural network
        %
        % treebagger - return a tree bagger 
        %
        Model (1,1) string ...
            {mustBeMember(Model,[...
                "automl"
                "selectml"
                "stackml"
                "tree"
                "svm"
                "ensemble"
                "gp"
                "linear"
                "kernel"
                "nnet"
                "treebagger"]) } = "automl"
    end %properties
    
    properties (Constant)
       Type = "Regression"
    end
    
    properties ( SetAccess = private, Hidden )
        ValidModelParameters = [
            "Learners"
            "OptimizeHyperparameters"
            "HyperparameterOptimizationOptions"
            "CrossValidation"
            "KFold"
            "Holdout"
            "Seed"
            ]
    end
    
    methods
        function obj = Regression(options)
            %experiment.Regression constructor
            
            arguments
               options.Data table
               options.DataFcn (1,1) function_handle {mustBeClass(options.DataFcn,"function_handle")}
               options.Model string {mustBeMember(options.Model,[...
                "automl" "selectml" "stackml" "tree" "svm", ...
                "ensemble" "gp" "linear" "kernel" "nnet" "treebagger"]) }
               options.DataConfiguration (1,:) optimizeParameter
               options.ModelConfiguration (1,:) optimizeParameter
               options.Description (1,1) string
               options.Name (1,1) string
               options.Search (1,1) string {mustBeMember(options.Search, ["gridsearch", "randomsearch"])} = "gridsearch"
               options.Evaluations (1,1) double {mustBeInteger}
               options.NumGridDivisions (1,1) double {mustBeInteger}
               options.CustomProperties (1,:) string 
               options.UseParallel (1,1) logical
            end
            
            args = namedargs2cell( options );
            
            obj = obj@experiment.Base( args{:} );

        end %experiment.Regression
        
    end %constructor
    
    methods  
        
        function fit(obj)
            %FIT Train specified ml for items in items/configuration in the
            %experiment. Note the data preparation via obj.prepare() must occur
            %prior to fit.
            %
            % Syntax:
            %
            % obj.fit() train specified ml for all items in the experiment.
            %
            
            %Validate Model Parameters
            obj.validateModel();
            
            if ~isempty( obj.Trials )
                obj.resetmodels()
                model = obj.Model;
                
                switch model
                    case {"stackml", "automl", "selectml"}
                        obj.( model );
                    otherwise
                        obj.instanceml();
                end
                
            end %if ~isempty
            
        end %function
        
        
        function [value, trials, models] = sort( obj, options )
            %SORT Rank experiment ML model results by an evaluation metric
            %
            % Syntax:
            %
            % value = obj.sort() rank model results on cross-validated mean
            % square error (mseOnCV).
            %
            % [value, runId, modelId] = obj.sort() return a ranked table
            % and corresponding ID for the experiment item and model
            % result.
            %
            % [value, ...] = obj.sort("PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Metric" - evaluation metric used to rank model results.
            %   Valid options are: mseOnCV, mseOnTest, or mseOnResub
            %
            
            arguments
                obj
                options.Metric (1,1) string ...
                    {mustBeMember(options.Metric,["mseOnCV", "rmseOnCV" "mseOnTest", "rmseOnTest" "mseOnResub" "rmseOnResub"])} = "mseOnCV"
            end
            
            args  = namedargs2cell( options );
            
            [value, trials, models] = sort@experiment.Base( obj, args{:} );
            
        end %sort
  
        
        function predictactual( obj, options )
            %PREDICTACTUAL Scatter plot of prediction versus actual (true)
            %values for all experimental runs. 
            %
            % Syntax:
            %
            % obj.predictactual() scatter plot with defaults. 
            %
            % obj.predictactual("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %
            %   "Item" - plot selected items by id (get id from
            %   obj.describe)
            %
            %   "Partition" - specify a partition. Valid options
            %   are "test" (default), "train", or "none".
            %
            
            arguments
                obj
                options.Partition (1,1) string ...
                    {mustBeMember( options.Partition, ["", "Test", "Train"] )}= "Test"
                options.Item (1,:) double = 1:height( obj.describe() );
            end %arguments 
            
             if ~isempty( obj.validTrialIndices )
            
                 h = tiledlayout( "flow", "TileSpacing", "compact" );
                 h.Parent.Color = "White";
                 
                 if options.Partition == "Test"
                     value = 0;
                 elseif options.Partition == "Train"
                     value = 1;
                 end
                 
                 [ trials, models ] = obj.ind2sub( options.Item );
                 
                 
                 for iItem = 1 : numel( options.Item )
                     
                     iTrial = trials( iItem );
                     iModel      = models( iItem );
                     
                     trial       = obj.Trials( iTrial );
                     data        = trial.Data;
                     actual      = trial.response;
                     prediction  = ("Prediction" + iModel);
                     
                     partitionname = "";
                     
                     if isprop( trial.Data.Properties.CustomProperties, "TrainingObservations" ) && options.Partition ~=""
                         
                         tF = trial.Data.Properties.CustomProperties.TrainingObservations == value;
                         partitionname = " ["+lower(options.Partition)+"]";
                         %Corner case
                         if all( tF  == 0 )
                             tF = true( height(table), 1 );
                             options.Partition = "";
                         end %if all(tF) == 0
                         
                     else
                         tF = true( height(data), 1 );
                     end %if isprop
                     
                     modeltype = extractBetween( trial.model(iModel).metadata.modelType,'(',')');
                     titlename = "trial "+ iTrial + ": " + modeltype + partitionname;
  
                     if iItem == 1
                         legendstate = true;
                     else
                         legendstate = false;
                     end

                     viz.predictactual(data(tF,:), actual, prediction, ...
                            "Axes", nexttile(), ...
                            "Title", titlename, ...
                            "Legend", legendstate );
                     
                 end %for iItem

             end %if isempty( obj.validTrialIndices )

        end %predictactual
        
        
        function plotmetric( obj, options )
            %PLOTMETRIC Plot evulation metric for all experimental runs as a
            %bar graph. 
            %
            % Syntax:
            %
            % obj.plotmetric()
            %
            % obj.plotmetric("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %   "Metric" - Evaluation critera used in visualization.
            %              Specify as a scalar string. Valid options
            %              are:   "mseOnCV", "mseOnResub", "mseOnTest", ...
            %                     "r2OnCV", "r2OnResub","r2OnTest", ...
            %                     "rmseOnCV", "rmseOnResub", "rmseOnTest".
            %
            %   "Sort"  - Sort experimental runs by evaluation criteria.
            %             Specify as a logical (default = true).
            %
            
            arguments
                obj           
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    ["mseOnCV", "mseOnResub", "mseOnTest", ...
                    "r2OnCV", "r2OnResub", "r2OnTest",...
                    "rmseOnCV", "rmseOnResub", "rmseOnTest"])} = "mseOnCV"
                options.Sort = true
                options.Axes = gobjects(0)
            end
            
            if isempty(options.Axes) && isgraphics(gca, 'axes')
                options.Axes = gca();
                
            % handles Chart case    
            elseif isempty(options.Axes) && ~isgraphics(gca, 'axes')
                figure("Color", "W");
                options.Axes = gca();
                
            else
                % Do nothing
            end
            
            args  = namedargs2cell( options );
            
            plotmetric@experiment.Base( obj, args{:} )
            
        end %plotmetric

    end %public methods
 
end %classdef

function mustBeClass(a,b)
    if class(a) ~= b
        error('Invalid class')
    end
end %function 
