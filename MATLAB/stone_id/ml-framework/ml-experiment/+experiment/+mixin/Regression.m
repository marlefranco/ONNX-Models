classdef Regression < experiment.mixin.Base
    %experiment.mixin.Regression ML Regression methods
    %
    % Copyright 2021 The MathWorks Inc.

    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, [ "linear" "tree" "svm", ...
            "ensemble" "gp" "kernel" "nnet" "treebagger" ...
            "all" "auto" "alllinear" "allnonlinear" "n/a"])};
    end %properties

    properties (Hidden, Constant)
        ValidAutoML = [ "tree" "linear" "svm", ...
            "ensemble" "gp"  "kernel" "nnet" "treebagger" ];
    end %properties

    methods (Access = protected)

        function validLearners(obj, trial, list)
            %validLearners

            arguments
                obj
                trial
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

            errorstack = string( {ME.stack.name} );

            if isempty(trial.model)
                trial.model = mdl;
            elseif any( contains(errorstack, "fitr.predictandupdate") )
                if ~isempty( trial.model(end).metadata )
                    mdl.testmetadata.modelType = trial.model(end).metadata.modelType;
                    trial.model(end).testmetadata = mdl.testmetadata;
                else
                    trial.model(end) = mdl;
                end
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(trial.Model)

        end %function

    end %methods

    %% Protected Methods
    methods (Access = protected)
        function iterations = fillFutureValues(~, models, trial, allmodelsettings)
            %Fill futures with parallel tasks
            for ii = 1:numel(models)

                fprintf( "Queuing: %s\n", models(ii) )
                iterations(ii) = parfeval(@experiment.Regression.trainModels,3,models(ii),trial,allmodelsettings);  %#ok<AGROW>

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
                            [mdl, info]  = fitr.net(trial.Data, allmodelsettings{:});
                        else
                            [mdl, info]  = fitr.nnet(trial.Data, allmodelsettings{:});
                        end
                    case "selectml"
                        if ver >= 2020
                            [mdl, info]  = fitr.auto(trial.Data, allmodelsettings{:});
                        else
                            [mdl, info]  = fitr.autointernal(trial.Data, allmodelsettings{:});
                        end

                    case "stackml"
                        [mdl, info]  = fitr.autointernal(trial.Data, allmodelsettings{:});

                    otherwise
                        [mdl, info]  = fitr.( iModel )(trial.Data, allmodelsettings{:});

                end

                thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                trial.model = [trial.model thismodel];

                [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                    trial.lastmodel().metadata.modelType, "ResponseName", trial.response );

                trial.Data = value;

                if ~isempty(testinfo)
                    trial.lastmodel().testmetadata = testinfo;
                end

            catch ME

                thismodel = experiment.Model.default( "ModelType", "Regression" );
                testinfo = thismodel.testmetadata;

                errorstack = string( {ME.stack.name} );

                if isempty(trial.model)
                    trial.model = thismodel;
                elseif any( contains(errorstack, "fitr.predictandupdate") )
                    if ~isempty( trial.model(end).metadata )
                        thismodel.testmetadata.modelType = trial.model(end).metadata.modelType;
                        trial.model(end).testmetadata = thismodel.testmetadata;
                    else
                        trial.model(end) = thismodel;
                    end
                else %Everything else... model or generic error
                    trial.model = [trial.model thismodel];
                end %if isempty(trial.Model)

                value = table();

            end %try/catch

        end %function

    end % static methods

end %classdef
