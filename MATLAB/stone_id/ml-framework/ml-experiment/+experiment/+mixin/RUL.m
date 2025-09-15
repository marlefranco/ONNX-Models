classdef RUL < experiment.mixin.Base
    %experiment.mixin.RUL PDM remaining useful life methods
    %
    % Copyright 2021 The MathWorks Inc.
    
    properties (Hidden, SetAccess = protected)
       Learners (1,:) string ...
           {mustBeMember(Learners, ["linDegradation" "expDegradation" "all" "n/a"])} 
    end %properties
    
    properties (Hidden, Constant)
        ValidAutoML = [ "linDegradation" "expDegradation" ];
    end %properties 
    
    methods (Access = protected)   
        
        function validLearners(obj, item, list)
            %validLearners
            
            arguments
                obj
                item
                list (1,:) string = ""
            end
            
            if list ~= ""
                obj.Learners = intersect( obj.ValidAutoML, list, 'stable' );
            else
                obj.Learners = obj.ValidAutoML;
            end
            
        end %function
        
        function updateLearners( obj, value )
            %updateLearners
            obj.Learners = value;
        end %function
        
        function handleError( obj, ME, trial )
            %handleErrorAutoML
            
            arguments
                obj
                ME
                trial (1,1) experiment.item.Supervised
            end %arguments
            
            warning(ME.identifier, '%s', ME.message)
            
            mdl = experiment.Model.default( "ModelType", obj.Type );
                        
            if isempty(trial.model)
                trial.model = mdl;
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(pipe.Model)
            
        end %function

        function handleParallelTraining(obj, models, trial, allmodelsettings)
            % handleParallelFutures Handle parallel future training
            
            % Fill futures
            iterations = fillFutureValues(obj, models, trial, allmodelsettings);
            
            for ii = 1:numel(models)
                trial.model = [trial.model, experiment.Model()];
            end

            for ii = 1:numel(models)
                
                % fetchNext blocks until next results
                [idx, model] = fetchNext(iterations);
                
                trial.model(idx) = model;
                
            end
        end
        
        function iterations = fillFutureValues(~, models, trial, allmodelsettings)
            %Fill futures with parallel tasks
            for ii = 1:numel(models)

                fprintf( "Queuing: %s\n", models(ii) )
                iterations(ii) = parfeval(@experiment.RUL.trainModels,1,models(ii),trial,allmodelsettings);  %#ok<AGROW>

            end
            disp('Running Models');
        end %function
    end %protected
    
    %% Static Methods

     methods (Static, Access = public)
        function [thismodel] = trainModels(iModel, trial, allmodelsettings)
            
            arguments
               iModel (1,1) string
               trial
               allmodelsettings
                
            end

            %Train and predict
            try
            switch iModel
                case "selectml"
                    [mdl, info]  = fitdegradation.auto(trial.Data, allmodelsettings{:});
                otherwise
                    [mdl, info]  = fitdegradation.( iModel )(trial.Data, allmodelsettings{:});
            end

            thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
            trial.model = [trial.model thismodel] ;

            catch
                thismodel = experiment.Model.default( "ModelType", "PDM" );
                        
                if isempty(trial.model)
                    trial.model = thismodel;
                else %Everything else... model or generic error
                    trial.model = [trial.model thismodel];
                end %if isempty(pipe.Model)
            end %try/catch
            
        end %function

    end % static methods

end %classdef

