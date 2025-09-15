classdef SemiSupervised < experiment.mixin.Base
    %experiment.mixin.SemiSupervised ML SemiSupervised Classification methods
    %
    % Copyright 2021 The MathWorks Inc.

    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, ["graph" "self" "all" "n/a"])}
    end %properties

    properties (Hidden, Constant)
        ValidAutoML = [ "graph" "self" ];
    end %properties

    methods (Access = protected)

        function automl( obj)
            %AUTOML

            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i);

                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "automl" );

                    %Append unlabeled data
                    unlabeled = trial.Data( ~trial.trainingobs, :);
                    settings = [{unlabeled}, allmodelsettings];

                    %Loop over Learners
                    models = obj.Learners;

                    %Train and Predict
                    obj.autoMLTraining(models, trial, settings);


                catch ME

                    obj.handleError( ME, trial )

                end %try/catch
            end %for iTrial

        end %function

        %TODO : selectml
        
        function instanceml( obj )
            %INSTANCEML Handles all single model training
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Append unlabeled data
                    unlabeled = trial.Data( ~trial.trainingobs, :);
                    settings = [{unlabeled}, allmodelsettings];
                    
                    %Train and predict
                    obj.trainModels(obj.Model, trial, settings);
                    
                catch ME
                   
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
        end %function

        function validLearners(obj, item, list)
            %validLearners

            arguments
                obj
                item %#ok<INUSA>
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
            %handleError

            arguments
                obj
                ME
                trial (1,1) experiment.item.Supervised
            end %arguments

            warning(ME.identifier, '%s', ME.message)

            responsename = trial.response;

            response = trial.Data.( responsename );
            if iscategorical( response )
                classes = string( categories( response ) );
            else
                classes = string( categories( categorical( response ) ) );
            end

            mdl = experiment.Model.default( "ModelType", obj.Type, ...
                "Classes", classes );

            errorstack = string( {ME.stack.name} );

            if isempty(trial.Model)
                trial.model = mdl;
            elseif any( contains(errorstack, "fits.predictandupdate") )
                mdl.testmetadata.modelType = trial.model(end).metadata.modelType;
                trial.model(end).testmetadata = mdl.testmetadata;
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(trial.model)

        end %function

        function iterations = fillFutureValues(~, models, trial, allmodelsettings)
            %Fill futures with parallel tasks
            for ii = 1:numel(models)

                fprintf( "Queuing: %s\n", models(ii) )
                iterations(ii) = parfeval(@experiment.SemiSupervised.trainModels,3,models(ii),trial,allmodelsettings);  %#ok<AGROW>

            end
            disp('Running Models');
        end %function


    end %methods

    %% Static Methods

    methods (Static, Access = public)
        function [thismodel, value, testinfo] = trainModels(iModel, trial, allmodelsettings)

            arguments
                iModel (1,1) string
                trial
                allmodelsettings

            end

            try
                [mdl, info]  = fits.( iModel )(trial.Data, allmodelsettings{:});

                thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                trial.model = [trial.model thismodel] ;

                [value, testinfo]  = fits.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                    trial.lastmodel().metadata.modelType );

                trial.Data = value;

                if ~isempty(testinfo)
                    trial.lastmodel().testmetadata = testinfo;
                end

            catch ME

                responsename = trial.response;

                response = trial.Data.( responsename );
                if iscategorical( response )
                    classes = string( categories( response ) );
                else
                    classes = string( categories( categorical( response ) ) );
                end

                thismodel = experiment.Model.default( "ModelType", "SemiSupervised", ...
                    "Classes", classes );
                testinfo = thismodel.testmetadata;

                errorstack = string( {ME.stack.name} );

                if isempty(trial.Model)
                    trial.model = thismodel;
                elseif any( contains(errorstack, "fits.predictandupdate") )
                    thismodel.testmetadata.modelType = trial.model(end).metadata.modelType;
                    trial.model(end).testmetadata = thismodel.testmetadata;
                else %Everything else... model or generic error
                    trial.model = [trial.model thismodel];
                end %if isempty(trial.model)

                value = table();

            end %try/catch

        end %function
    end

end %classdef

