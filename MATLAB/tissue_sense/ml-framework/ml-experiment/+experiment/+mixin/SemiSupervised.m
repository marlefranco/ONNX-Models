classdef SemiSupervised < experiment.mixin.Base
    %experiment.mixin.SemiSupervised ML SemiSupervised Classification methods
    %
    
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
                for iModel = models(:)'
                    try
                        fprintf( "Running: %s\n", iModel )
                        
                        [mdl, info]  = fits.( iModel )(trial.Data, settings{:});
                        
                        thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                        trial.model = [trial.model thismodel] ;
                        
                        [value, testinfo]  = fits.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                            trial.lastmodel().metadata.modelType );
                        
                        trial.Data = value;
                        
                        if ~isempty(testinfo)
                            trial.lastmodel().testmetadata = testinfo;
                        end
                        
                        
                    catch ME
                        
                        obj.handleError( ME, trial )
                        
                    end %try/catch 
                end %for iModel
                
            end %for iTrial
            
        end %function
        
        %TODO : selectml
        
        function graph( obj )
            %GRAPH Fit a graph based semi supervised model in trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i);
                                        
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Append unlabeled data 
                    unlabeled = trial.Data( ~trial.trainingobs, :);
                    settings = [{unlabeled}, allmodelsettings];
                    
                    %Train and Predict
                    [mdl, info]  = fits.graph(trial.Data, settings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fits.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                        trial.lastmodel().metadata.modelType );
                    
                    trial.Data = value;
                    
                    if ~isempty(testinfo)
                        trial.lastmodel().testmetadata = testinfo;
                    end
                    
                catch ME
                    
                   obj.handleError( ME, trial )
                   
                end %try/catch 
            end %for iTrial
            
        end %graph
        
        
        function self( obj )
            %SELF Fit a self training semi supervised model in trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i);
                 
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Append unlabeled data
                    unlabeled = trial.Data( ~trial.trainingobs, :);
                    settings = [{unlabeled}, allmodelsettings];
                    
                    %Train and predict
                    [mdl, info]  = fits.self(trial.Data, settings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fits.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                        trial.lastmodel().metadata.modelType );
                    
                    trial.Data = value;
                    
                    if ~isempty(testinfo)
                        trial.lastmodel().testmetadata = testinfo;
                    end
                    
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
            
            classes = string( categories( trial.Data.( responsename ) ) );
            
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
        
        
    end %methods
end %classdef

