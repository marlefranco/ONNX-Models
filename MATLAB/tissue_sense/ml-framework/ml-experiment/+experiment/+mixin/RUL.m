classdef RUL < experiment.mixin.Base
    %experiment.mixin.RUL PDM remaining useful life methods
    %
    
    properties (Hidden, SetAccess = protected)
       Learners (1,:) string ...
           {mustBeMember(Learners, ["linDegradation" "expDegradation" "all" "n/a"])} 
    end %properties
    
    properties (Hidden, Constant)
        ValidAutoML = [ "linDegradation" "expDegradation" ];
    end %properties 
    
    methods (Access = protected)
        
        function automl( obj )
            
            for i = 1:numel( obj.Trials )
                
                %Get item
                trial = obj.Trials(i);
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "automl" );
        
                %Loop over Learners
                models = obj.Learners;
                
                %Train
                for iModel = models(:)'
                    try
                        fprintf( "Running: %s\n", iModel )                        
                        [mdl, info]  = fitdegradation.( iModel )(trial.Data, allmodelsettings{:});
                           
                        thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                        trial.model = [trial.model thismodel] ;
 
                    catch ME
                        
                        obj.handleError( ME, trial )
                        
                    end
                end %for iModel
                
            end %for iTrial
    
        end %function
        
        
        function selectml( obj )
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "selectml" );
                    
                    %Train
                    fprintf( "Running: selectml fitdegradation.auto\n")
                    [mdl, info]  = fitdegradation.auto(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
        function linDegradation( obj )
            %linearDegradationModel pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);

                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train
                    [mdl, info]  = fitdegradation.linDegradation(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %linDegradation
        
        
        function expDegradation( obj )
            %exponentialDegradationModel pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);

                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train
                    [mdl, info]  = fitdegradation.expDegradation(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                catch ME
                    
                   obj.handleError( ME, trial )
                
                end %try/catch
            end %for iTrial
            
        end %function
        
        
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
                trial (1,1) experiment.trial.Supervised
            end %arguments
            
            warning(ME.identifier, '%s', ME.message)
            
            mdl = experiment.Model.default( "ModelType", obj.Type );
                        
            if isempty(trial.Model)
                trial.model = mdl;
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(pipe.Model)
            
        end %function
        
    end %protected
end %classdef

