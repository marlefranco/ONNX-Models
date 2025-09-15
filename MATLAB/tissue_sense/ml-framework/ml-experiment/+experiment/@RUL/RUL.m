classdef RUL < experiment.Base & experiment.mixin.RUL
    %RUL Summary of this class goes here
    
    
    properties
        Model (1,1) string ...
            {mustBeMember(Model,[...
            "automl"
            "selectml"
            "linDegradation"
            "expDegradation"])} = "automl"
    end
    
    properties (Constant)
        Type = "PDM"
    end
    
    properties ( SetAccess = private, Hidden )
        ValidModelParameters = [
            "Learners"
            "HealthIndicatorName"
            "DataVariable"
            "LifeTimeVariable"
            "LifeTimeUnit"
            "UseParallel"
            ]
    end
        
    
    methods
        function obj = RUL(varargin)
            %RUL Construct an instance of this class
            
            obj = obj@experiment.Base( varargin{:} );
            
        end %constructor
        
        
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
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    ["r2OnTrain","rmseOnTrain"])} = "rmseOnTrain"
            end
            
            args  = namedargs2cell( options );
            
            [value, trials, models] = sort@experiment.Base( obj, args{:} );
            
        end %sort
        
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
            %              are:   "r2OnTrain","rmseOnTrain".
            %
            %   "Sort"  - Sort experimental runs by evaluation criteria.
            %             Specify as a logical (default = true).
            %
            
            arguments
                obj
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    ["r2OnTrain","rmseOnTrain"])} = "rmseOnTrain"
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
        
    end %methods
    
    
    methods ( Access = protected )
        
        function i_postprepareChecks( obj )
            
            for  i = 1:numel( obj.Trials )
                
                trial = obj.Trials(i);
                
                %Validate standard custom properties definition in preparation
                %pipeline
                tf_HealthIndicator  = isprop( trial.Data.Properties.CustomProperties, "HealthIndicatorName" );
                tf_DataVariable     = isprop( trial.Data.Properties.CustomProperties, "DataVariable" );
                tf_LifeTimeVariable = isprop( trial.Data.Properties.CustomProperties, "LifeTimeVariable" );
                tF_Options          = isprop( trial.Data.Properties.CustomProperties, "Options" );
                 
                if ~tf_HealthIndicator
                    warning( ['HealthIndicator is undefined in data preparation pipeline. ' ...
                        'Please see newPipelineTemplate("byType", "RUL") for example of defining property list.'])
                end %if tF_HealthIndicator
                
                if ~tf_DataVariable
                    warning( ['DataVariable is undefined in data preparation pipeline. ' ...
                        'Please see newPipelineTemplate("byType", "RUL") for example of defining property list.'])
                end %if tf_DataVariable
                
                if ~tf_LifeTimeVariable
                    warning( ['LifeTimeVariable is undefined in data preparation pipeline. ' ...
                        'Please see newPipelineTemplate("byType", "RUL") for example of defining property list.'])
                end %if tf_LifeTimeVariable
                
                if ~tF_Options
                      warning( ['Pipeline options are not stored in custom properties. ' ...
                        'If undefined, you will not have a record of the options used to configure experiment trial.'...
                        'Please see newPipelineTemplate for example of storing pipeline options.'])
                end %if tF_Options
                
                %TODO Post pipeline validation checks for RUL
                
                %                 %Validate data
                %                 if tf_Feature
                %                     features = pipe.Data.Properties.CustomProperties.Features;
                %                 else
                %                     features = string( pipe.Data.Properties.VariableNames(1:end-1) );
                %                 end %if tf_Feature
                %
                %                 datasummary = util.summarize( pipe.Data(:, features));
                %
                %                 if datasummary.tF_Missing
                %                     warning( 'Training data contain NaN.' )
                %                 end
                %
                %                 if datasummary.tF_Constant
                %                     warning( 'Training data contain constant variables.' )
                %                 end
                %
                %                 if datasummary.tF_ConstantInClass == true
                %                     warning( 'Training data contain constant variables in one or more response classes.' )
                %                 end
                %
            end %for iTrial
            
        end %function
        
    end %methods
end %classdef
