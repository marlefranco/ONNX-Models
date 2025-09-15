classdef Classification < experiment.mixin.Base
    %experiment.mixin.Classification ML Classification methods
    %
    % Copyright 2021 The MathWorks Inc.

    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, ["tree" "discr" "nb" "knn" "svm" "linear",...
            "kernel" "ensemble" "ecoc" "nnet" "treebagger", ...
            "all" "auto" "all-linear" "all-nonlinear" "n/a"])}
    end %properties


    properties (Hidden, Constant)
        ValidTwoClass = ["tree", "discr", "nb","knn", "svm", "linear",...
            "kernel", "ensemble", "ecoc", "nnet", "treebagger"];

        ValidMultiClass = ["tree", "discr", "nb", "knn",...
            "ensemble", "ecoc", "nnet", "treebagger"]
    end %properties


    methods (Access = protected)

        function stackml( obj )
            for i = 1:numel( obj.Trials )

                %Get item
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

                %Initialize settings for meta model
                allmodelsettings = obj.initializeModelParams( trial, "stackml" );
                try
                    %Run meta model
                    fprintf( "Running: stackml\n")

                    % Make sure learners are intialized based on 2 vs 3
                    % class valid models
                    obj.validLearners(trial);
                    learnStruct.Learners = obj.Learners;
                    allmodelsettings = [allmodelsettings, namedargs2cell( learnStruct )]; %#ok<AGROW>

                    %Train and predict
                    obj.trainModels("stackml", trial, allmodelsettings);

                    trial.lastmodel().metadata.modelType = trial.lastmodel().metadata.modelType + " Stack";
                    trial.lastmodel().testmetadata.modelType = trial.lastmodel().testmetadata.modelType + " Stack";

                catch ME

                    obj.handleError( ME, trial )

                end %try/catch

            end %for iTrial
        end %stackml

        function validLearners( obj, item, list )
            %validLearners

            arguments
                obj
                item experiment.item.Base
                list (1,:) string = ""
            end

            response = item.response;

            nClasses = numel( unique(item.Data.(response) ) );

            if nClasses < 3
                obj.Learners = obj.ValidTwoClass;
            else
                obj.Learners = obj.ValidMultiClass;
            end

            if list ~= ""
                obj.Learners = intersect( obj.Learners, list, 'stable' );
                if isempty(obj.Learners)
                    error(list+" model is not compatible with MultiClass problems")
                end
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

            if isempty(trial.model)
                trial.model = mdl;
            elseif any( contains(errorstack, "fitc.predictandupdate") )
                mdl.testmetadata.modelType = trial.model(end).metadata.modelType;
                trial.model(end).testmetadata = mdl.testmetadata;
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(trial.Model)

        end %function

    end %protected

    %% Protected Methods
    methods (Access = protected)
        function iterations = fillFutureValues(~, models, trial, allmodelsettings)
            %Fill futures with parallel tasks
            for ii = 1:numel(models)

                fprintf( "Queuing: %s\n", models(ii) )
                iterations(ii) = parfeval(@experiment.Classification.trainModels,3,models(ii),trial,allmodelsettings);  %#ok<AGROW>

            end
            disp('Running Models');
        end %function
    end

    %% Static Methods
    methods (Static, Access = public)
        function [thismodel, value, testinfo] = trainModels(iModel, trial, allmodelsettings)

            arguments
                iModel (1,1) string
                trial
                allmodelsettings

            end

            %Version flag
            ver = str2double(extractBetween(string(version),"R", ("a"|"b")));

            %Train and predict
            try
                switch iModel
                    case "nnet"
                        if  ver >= 2021
                            [mdl, info]  = fitc.net(trial.Data, allmodelsettings{:});
                        else
                            [mdl, info]  = fitc.nnet(trial.Data, allmodelsettings{:});
                        end
                    case "selectml"
                        if ver >= 2020
                            [mdl, info]  = fitc.auto(trial.Data, allmodelsettings{:});
                        else
                            [mdl, info]  = fitc.autointernal(trial.Data, allmodelsettings{:});
                        end

                    case "stackml"
                        [mdl, info]  = fitc.autointernal(trial.Data, allmodelsettings{:});

                    otherwise
                        [mdl, info]  = fitc.( iModel )(trial.Data, allmodelsettings{:});

                end

                thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                trial.model = [trial.model thismodel] ;

                [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                    trial.lastmodel().metadata.modelType, "ResponseName", trial.response );

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

                thismodel = experiment.Model.default( "ModelType", "Classification", ...
                    "Classes", classes );
                testinfo = thismodel.testmetadata;

                errorstack = string( {ME.stack.name} );

                if isempty(trial.model)
                    trial.model = thismodel;
                elseif any( contains(errorstack, "fitc.predictandupdate") )
                    thismodel.testmetadata.modelType = trial.model(end).metadata.modelType;
                    trial.model(end).testmetadata = thismodel.testmetadata;
                else %Everything else... model or generic error
                    trial.model = [trial.model thismodel];
                end %if isempty(trial.Model)

                value = table();

            end %try/catch

        end %function

    end

end %classdef
