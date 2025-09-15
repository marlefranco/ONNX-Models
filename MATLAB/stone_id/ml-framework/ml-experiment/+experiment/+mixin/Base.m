classdef Base < matlab.mixin.SetGet
    %experiment.mixin.Base 
    %  
    % Copyright 2021 The MathWorks Inc.
    %% Abstract Methods
    methods (Abstract, Access = protected)
         fillFutureValues
    end
    
    %% Model Training Methods
    methods (Access = protected)
        function automl( obj )
            %AUTOML
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "automl" );
                    
                    %Loop over Learners
                    models = obj.Learners;
                    
                    % Train and Predict
                    obj.autoMLTraining(models, trial, allmodelsettings);
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end % try/catch
                
            end %for iTrial
            
        end %function
        
        function selectml( obj )
            %SELECTML
            
            for i = 1:numel( obj.Trials )
                
                %Get trial
                trial = obj.Trials(i); % trial = trial.result;
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "selectml" );
                
                %Train and predict
                try
                    fprintf( "Running: selectml fitrauto\n")
                    obj.trainModels("selectml", trial, allmodelsettings);
                                        
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch
            end %for iTrial
            
        end %function
        
        function stackml( obj )
            
            for i = 1:numel( obj.Trials )
                
                %Get trial
                trial = obj.Trials(i);
                
                %Run automl
                allmodelsettings = obj.initializeModelParams( trial, "automl" );
                
                %Loop over Learners
                models = obj.Learners;
                
                % Train and Predict
                try 
                    obj.autoMLTraining(models, trial, allmodelsettings);

                catch ME

                    obj.handleError( ME, trial )

                end % try/catch
                
                try
                    
                    %Initialize settings for meta model
                    allmodelsettings = obj.initializeModelParams( trial, "stackml" );
                    
                    %Run meta model
                    fprintf( "Running: stackml\n")

                    %Train and predict
                    obj.trainModels("stackml", trial, allmodelsettings);
                    
                    trial.lastmodel().metadata.modelType = trial.lastmodel().metadata.modelType + " Stack";
                    trial.lastmodel().testmetadata.modelType = trial.lastmodel().testmetadata.modelType + " Stack";

                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch
            end %for iTrial
            
        end %function
        
        function instanceml( obj )
            %INSTANCEML Handles all single model training
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    if matches(class(obj), "experiment.Classification")
                        %Validate binary
                        nClasses = numel( unique(trial.Data.(trial.response) ) );
                        if nClasses > 2
                            return
                            %TODO add warning
                        end
                    end
                    
                    %Train and predict
                    obj.trainModels(obj.Model, trial, allmodelsettings);
                    
                catch ME
                   
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
        end %function
        
    end % protected methods
      
    %%
    methods (Access = protected)
    
        function validLearners(  obj )   %#ok<MANU>
        end %function
        
        function updateLearners(  obj )    %#ok<MANU>
        end %function
        
        function handleError( obj ) %#ok<MANU>
        end %function
        
        function value = initializeModelParams( obj, trial, mthd )
            %initializeModelParams
            
            arguments
                obj
                trial experiment.item.Base
                mthd (1,1) string {mustBeMember(mthd, ["automl", "selectml", "stackml", "fit"])}
            end
            
            %Initialize item model parameters
            if obj.Type == "PDM" 
                value = trial.initializeModelParams( "pdm" );
            else
                value = trial.initializeModelParams();
            end %if obj.Type
                        
            %Parse settings for learner property (extract then remove)
            switch mthd
                case "automl"
                    [value, learners] = parseAndExtractArgs( value, "Learners" );
                    
                    try
                        %If Model configuration exists, else use all valid learners
                        if ~isempty( value )
                            
                            %If Learners are specified, else use all valid
                            %learners
                            if ~isempty( learners )
                                
                                if any( matches( learners, "all") )
                                    obj.validLearners( trial );
                                else
                                    obj.validLearners( trial, learners);
                                end %if any( matches...
                                
                            else
                                obj.validLearners( trial );
                            end %if ~isempty( learners)
                        else
                            obj.validLearners( trial );
                        end %if ~isempty( value )
                        
                    catch ME
                        throw(ME)
                    end
                    
                case "selectml"
                    
                    learners = parseArgs( value, "Learners" );
                    
                    if ~isempty( learners )
                        obj.updateLearners( learners );
                    else
                        switch obj.Type
                            case "Classification"
                                obj.updateLearners( "auto" );
                            otherwise
                                obj.updateLearners( "all" );
                        end %switch 
                        
                    end %if ~isempty( learners)
                    
                case "stackml"
                    
                    %Override defaults (only applies to metamodel). No
                    %optimization on stacked model.
                    value = struct();
                    
                    %Use prediction as features, invert training set
                    vars     = trial.VariableNames;
                    response = trial.response;
                    features = vars(contains(vars, "Prediction"));
                    
                    value.PredictorNames = features;
                    value.ResponseName = response;
                    value.Include = ~trial.trainingobs;
                    value.Learners = obj.Learners;
                            
                    value = namedargs2cell( value );
                      
                case "fit"
                    
                    %If learners is supplied to standard fit* function, remove
                    value = parseAndExtractArgs( value, "Learners" );
                    obj.updateLearners( "n/a" );
                    
                otherwise
                    error("Unhandled option.")
            end %switch
                        
        end %function

        function autoMLTraining(obj, models, trial, allmodelsettings)
            % AUTOMLTRAINING Internal method to handle auto ML training and parallelization

            if obj.UseParallel
                %Train and Predict

                hyperOptFlag = false;
                index = zeros(1,length(allmodelsettings));
                for ii = 1:2:length(allmodelsettings)-1
                    index(ii) = strcmp(allmodelsettings{ii}, 'OptimizeHyperparameters');
                end

                if any(index)
                    tF = find(index) + 1;
                    hyperOptFlag = ~(allmodelsettings{tF(1)} == "none");
                end

                % check for hyperparameter optimization model setting
                if hyperOptFlag
                    for iModel = models(:)'

                        fprintf( "Running: %s\n", iModel )
                        obj.trainModels(iModel, trial, allmodelsettings);

                    end %for iModel
                else

                    % Set up execution environment
                    if isempty(gcp('nocreate'))
                        parpool()
                    end

                    % Handle parallel training 
                    obj.handleParallelTraining(models, trial, allmodelsettings);
                end
            else
                %Train and Predict
                for iModel = models(:)'

                    fprintf( "Running: %s\n", iModel )
                    obj.trainModels(iModel, trial, allmodelsettings);

                end %for iModel

            end %if UseParallel
        end % function

        function handleParallelTraining(obj, models, trial, allmodelsettings)
            % handleParallelFutures Handle parallel future training
            
            % Fill futures
            iterations = fillFutureValues(obj, models, trial, allmodelsettings);
            
            predictions = table();
            
            for ii = 1:numel(models)
                trial.model = [trial.model, experiment.Model()];
            end

            for ii = 1:numel(models)
                % fetchNext blocks until next results
                [idx, model, value, testinfo] = fetchNext(iterations);

                trial.model(idx) = model;

                if ~isempty(value)
                    prediction = value.("Prediction1");
                    varName = "Prediction" + (idx);

                    predictions.( varName ) = prediction;
                end

                if ~isempty(testinfo)
                    trial.model(idx).testmetadata = testinfo;
                end
            end
            
            trial.Data = horzcat( value(:, 1:end-1), predictions );
            
        end
    end %methods

    %% Static Methods
    methods (Static, Access = public)
        %Implement in subclass
        trainModels(iModel, trial, allmodelsettings)
    end

end %classdef 


%Local
function [value, splitvalue] = parseAndExtractArgs( value, name )
    %parseAndExtractArgs

    indx = cellfun(@(x)ismember(x,name), value(1:2:end) );

    if ~isempty( indx ) &&  any( indx )
        splitvalue = value{ find(indx)*2 };
        value( find(indx)*2-1:find(indx)*2 ) = [];
    else
        splitvalue = {};
    end %if ~isempty(indx)

end %function


function splitvalue = parseArgs( value, name )
    %parseArgs
    
    indx = cellfun(@(x)ismember(x,name), value(1:2:end) );

    if ~isempty(indx) &&  any( indx )
        splitvalue = value{ find(indx)*2 };
    else
        splitvalue = {};
    end %if ~isempty(indx)

end %function