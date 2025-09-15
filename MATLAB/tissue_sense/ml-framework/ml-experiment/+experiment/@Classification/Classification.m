classdef Classification < experiment.Base & experiment.mixin.Classification 
    %experiment.Classification Create a machine learning classification experiment. Each
    %run in an experiment consists of: (1) a unique data pipeline configuration,
    %(2) a selected ml model, and (3) a model/hyperparamter configuration.
    %
    % General Syntax:
    %  session = experiment.Classification(...
    %       "Data", thisData, ...
    %       "DataFcn", @(x, settings)thisPipeline(x, settings{:}), ...
    %       "Model", thisModel, ...
    %       "DataConfiguration", theseParameters, ...
    %       "ModelConfiguration", theseParameters, ...
    %       "Description, "Some description of experiment" )
    %
    % Description:
    %   experiment.Classification( "Data", dataset, ...
    %       "DataFcn", @(x, settings)datapipeline(x, settings{:} ) will
    %       create a classification experiment using the default pipeline. In a
    %       standard experiment the "Data" will be processed/prepared using
    %       the "DataFcn" specified above and automated machine learning
    %       (autoML) applied. By default, cross-validation is used (kFold =
    %       5) and hyperparameter optimization is not performed. To change
    %       these options specify "ModelConfiguration" parameters (see
    %       below).
    %       
    %       Data: is a table dataset 
    %       DataFcn: is an anonymous function specified in the format above
    %
    %   experiment.Classification( __, ...
    %       "Model", modelselection) specifies an optional model selection.
    %    
    %       Model: is a scalar string. Specify: "automl", "selectml",
    %       "stackml", or a learner from the classification model set
    %       below. Configure the model selection via the ModelConfiguration
    %       property described below.
    %       
    %       Automated Machine Learning 
    %           *automated ML (all classification models): "automl"
    %           *select ML ('best' classification model): "selectml"
    %           *stack ML (meta-model combining all classifications): "stackml"
    %           
    %       multi-class classification models
    %           *classification tree: "tree"
    %           *classification discriminant analysis: "discr"
    %           *classification naive bayes: "nb" 
    %           *classification k nearest: "knn"
    %           *classification ensemble: "ensemble" 
    %           *classification multi-class svm: "ecoc" 
    %           *classification tree bagger: "treebagger"
    %           *classification neural net: "nnet"
    %
    %       two-class classification models (n==2)
    %           *classification svm:  "svm"           
    %           *classification linear: "linear"
    %           *classification kernel: "kernel"
    %
    %   experiment.Classification( __, ...
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
    %           "Range", ["none" "fscchi2"],...
    %           "Type", "Discrete");
    %
    %       Valid parameters for MODELCONFIGURATION for all model
    %       selection include:
    %           - OptimizeHyperparameters
    %           - HyperparameterOptimizationOptions  
    %           - Cost
    %           - CrossValidation
    %           - KFold
    %           - HoldOut        
    %           - PredictorNames
    %           - ResponseNames
    %           - Include
    %
    %       Note that there are additional options specific to automated ML.
    %
    %       For automl and selectml these options include:
    %           - Learners
    %
    %       Example: Configure model with/without hyperparameter optimization
    %       modelConfiguration = optimizeParameter.new(...
    %       	"Name", "OptimizeHyperparameters", ...
    %           "Range", ["none" "all"],...
    %           "Type", "Discrete");
    %
    %       Refer to <a href="matlab:help('fitc')">fitc</a> documentation for general information on
    %       parameter usage, or <a href="matlab:help('fitc/tree')">fitc/tree</a> for a specific method.  
    %
    %   experiment.Classification( __, ...
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
    %   experiment.Classification methods:
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
    %   plotmetric      - visualize model error as bar graph
    %   confustionchart - visualize confusion charts
    %
    %   save          - export experiment items/models to disk  
    %   writedata     - export trial data results with predictions
    %
    %   experiment.Classification properties
    %   
    %   Data                    - Input data
    %   DataFcn                 - Specify data preparation pipeline 
    %   Model                   - Select classification method
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
    %   MathWorks 2020
    %
    
    properties
        %Model Select classification method. Either automated machine learning
        %or a supported classification method from list below. Note that these
        %options are further configured using the ModelConfiguration
        %property.
        %
        % automl(default) - return multiple classification models (default all). A
        % subset of models can be specified by setting "Learners" in the
        % ModelConfiguration. 
        %
        % selectml - run multiple classification methods and return the
        % optimal model based a specified evaluation criteria (default
        % errorOnCV). The evaluation criteria can be specified by setting
        % "Metric" in the ModelConfiguration.
        %
        % Multi-class models:
        %
        % tree - return classification trees 
        %
        % discr - return classification discriminant
        %
        % nb - return classification naive bayes
        %
        % knn - return classification k nearest
        %
        % ensemble - return classification ensemble
        %
        % ecoc - return multi-class classification svm  
        %  
        % nnet - return a neural network
        %
        % treebagger - return a tree bagger 
        %
        % Two-class models:
        %
        % svm - return two-class classificaiton support vector 
        %
        % linear - return a linear classification
        % 
        % kernel - return a kernel classification
        %
         Model (1,1) string ...
            {mustBeMember(Model,[...
                "automl"
                "selectml"
                "stackml"
                "tree"
                "discr"
                "nb"
                "knn"
                "svm"
                "ensemble"
                "ecoc"
                "linear"
                "kernel"
                "nnet"]) } = "automl" 

    end %properties
    
    properties (Constant)
        Type = "Classification"
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
        function obj = Classification(options)
            %experiment.Classification constructor
            
            arguments
                options.Data table
                options.DataFcn (1,1) function_handle {mustBeClass(options.DataFcn,"function_handle")}
                options.Model string {mustBeMember(options.Model,[...
                    "automl" "selectml" "stackml" "tree" "svm", ...
                    "ensemble" "gp" "linear" "kernel" "nnet" "treebagger"]) }
                options.DataConfiguration (1,:) optimizeParameter
                options.ModelConfiguration (1,:) optimizeParameter
                options.AdditionalConfiguration (1,:) constantParameter
                options.Description (1,1) string 
                options.Search (1,1) string {mustBeMember(options.Search, ["gridsearch", "randomsearch"])} = "gridsearch"
                options.Evaluations (1,1) double {mustBeInteger}
                options.NumGridDivisions (1,1) double {mustBeInteger}
                options.CustomProperties (1,:) string 
            end
            
            args = namedargs2cell( options );
            
            obj = obj@experiment.Base( args{:} );
            
        end %experiment.Classification
        
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
                obj.( model );
            end %if ~isempty
 
        end %function
        
        
        function [value, trials, models] = sort( obj, options )
            %SORT Rank experiment ML model results by an evaluation metric
            %
            % Syntax:
            %
            % value = obj.sort() rank model results on cross-validated
            % classification error (errorOnCV). 
            %
            % [value, runId, modelId] = obj.sort() return a ranked table
            % and corresponding ID for the experiment item and model
            % result.
            %
            % [value, ...] = obj.sort("PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Metric" - evaluation metric used to rank model results.
            %   Valid options are: errorOnResub, errorOnCV, errorOnTest,
            %   f1ScoreOnTrain, f1ScoreOnTest, AUCOnTrain, "AUCOnTest.
            %
            
            arguments
                obj
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    ["errorOnResub","errorOnCV", "errorOnTest", ...
                    "f1ScoreOnTrain","f1ScoreOnTest", ...
                    "AUCOnTrain","AUCOnTest"])} = "errorOnCV"
            end
            
            args  = namedargs2cell( options );
            
            [value, trials, models] = sort@experiment.Base( obj, args{:} );
            
        end %sort

        
        function plotmetric( obj, options )
            %PLOTMETRIC Plot evalution metric for all experimental runs as a
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
            %              are:   "errorOnResub","errorOnCV", "errorOnTest", ...
            %                     "f1ScoreOnTrain","f1ScoreOnTest", ...
            %                     "AUCOnTrain","AUCOnTest".
            %
            %   "Sort"  - Sort experimental runs by evaluation criteria.
            %             Specify as a logical (default = true).
            %
            
            arguments
                obj           
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    ["errorOnResub","errorOnCV", "errorOnTest", ...
                    "f1ScoreOnTrain","f1ScoreOnTest", ...
                    "AUCOnTrain","AUCOnTest"])} = "errorOnCV"
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
            
        end %function
        
        function confusionchart( obj, options )
            %CONFUSIONCHART Plot confusion chart for trial+models in the
            %experiment.
            %
            % Syntax:
            %   obj.confusionchart() defaults to use test set if available
            %
            %   obj.confusionchrt( "Partition", thisset ) specify set as
            %   ""(none), "Train", or "Test".
            %
            
            arguments
                obj
                options.Partition (1,1) string ...
                    {mustBeMember(options.Partition,["", "Train" "Test"])} = "Test"; 
                options.Item (1,:) double = 1:height( obj.describe() )
            end
            
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
                    titlename = "trial "+ iTrial + ": " + modeltype;
                    
                    nexttile()
                    viz.confusionchart( data(tF,:), actual, prediction, ...
                        "RowSummary", 'row-normalized', ...
                        "ColumnSummary", 'column-normalized', ...
                        "Title", titlename);
                    
                    if options.Partition == ""
                        cmtitle = "All Data";
                    else
                        cmtitle = options.Partition + " Set";
                    end
                    
                    sgtitle( cmtitle )
                    
                end %for iItem

            end %if ~isempty( obj.validTrialIndices )

        end %function
        
    end %public
    
    methods (Access = protected)
        function i_postprepareChecks( obj )
            
            i_postprepareChecks@experiment.Base( obj )
            
            %Special case 
             for  i = 1:numel( obj.Trials )
                trial = obj.Trials(i); 
                
                this = trial.response;
                
                if ~iscategorical( trial.Data.(this) )
                   warning( ['Response variable is not defined as categorical in data preparation pipeline. ' ...
                        'Response will be converted to categorical during fit. ' ...
                        'This may have unintended consequences if training partition does not include all classes. ' ...
                        'Best practice is to explicitly define resposne as categorical in data preparation pipeline.'])
                end
                  
             end
            
        end %function
    end %methods
    
end %experiment.Classification

function mustBeClass(a,b)
    if class(a) ~= b
        error('Invalid class')
    end
end %function 
        
