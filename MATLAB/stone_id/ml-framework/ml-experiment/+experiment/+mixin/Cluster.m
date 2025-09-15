classdef Cluster < experiment.mixin.Base
    %experiment.mixin.Cluster ML Clustering methods

    % Copyright 2021 The MathWorks Inc.

    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, ["hierarchical" "som" "kmeans", ...
            "kmedoids" "gmm" "spectral" "all" "n/a"])};
    end %properties


    properties (Hidden, Constant)
        ValidAutoML = [ "hierarchical" "som" "kmeans", ...
            "kmedoids" "gmm" "spectral" ];
    end %properties


    methods (Access = protected)

        %TODO : selectml

        function validLearners(obj, trial, list)
            %validLearners

            arguments
                obj
                trial %#ok<INUSA>
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
                trial (1,1) experiment.item.Unsupervised
            end %arguments

            warning(ME.identifier, '%s', ME.message)

            lbl = experiment.Label.default( "ModelType", obj.Type );

            if isempty(trial.Label)
                trial.Label = lbl;
            else %Everything else... model or generic error
                trial.Label = [trial.Label lbl];
            end %if isempty(trial.Model)

        end %function

        function handleParallelTraining(obj, models, trial, allmodelsettings)
            % handleParallelFutures Handle parallel future training

            % Fill futures
            iterations = fillFutureValues(obj, models, trial, allmodelsettings);

            labels = table();

            for ii = 1:numel(models)
                trial.Label = [trial.Label, experiment.Label()];
            end

            for ii = 1:numel(models)
                % fetchNext blocks until next results
                [idx, thisLabel, result] = fetchNext(iterations);

                trial.Label(idx) = thisLabel;

                if ~isempty(result)
                    label = result.("Label1");
                    varName = "Label" + (idx);

                    labels.( varName ) = label;
                end

            end

            trial.Data = horzcat( result(:, 1:end-1), labels );
        end

        function iterations = fillFutureValues(~, models, trial, allmodelsettings)
            %Fill futures with parallel tasks
            for ii = 1:numel(models)

                fprintf( "Queuing: %s\n", models(ii) )
                iterations(ii) = parfeval(@experiment.Cluster.trainModels,2,models(ii),trial,allmodelsettings);  %#ok<AGROW>

            end
            disp('Running Models');
        end %function

    end %protected

    %% Static Methods

    methods (Static, Access = public)
        function [thisLabel, result] = trainModels(iModel, trial, allmodelsettings)

            arguments
                iModel (1,1) string
                trial
                allmodelsettings

            end

            %Train and predict
            try
                [value, info] = clst.( iModel )(trial.Data, allmodelsettings{:});

                thisLabel = experiment.Label( 'label', value, 'metadata', info );

                trial.Label = [trial.Label thisLabel];

                result = clst.assignlabel( trial.Data, trial.lastfit().label );

                trial.Data = result;
            catch

                thisLabel = experiment.Label.default( "ModelType", "Unsupervised" );

                if isempty(trial.Label)
                    trial.Label = thisLabel;
                else %Everything else... model or generic error
                    trial.Label = [trial.Label thisLabel];
                end %if isempty(trial.Model)

                result = table();

            end %try/catch
        end %function
    end

end %experiment.trial.Unsupervised

