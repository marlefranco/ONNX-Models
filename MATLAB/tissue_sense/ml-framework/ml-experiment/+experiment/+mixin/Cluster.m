classdef Cluster < experiment.mixin.Base
    %experiment.mixin.Cluster ML Clustering methods
    
    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, ["hierarchical" "som" "kmeans", ...
            "kmedoids" "gmm" "spectral" "all" "n/a"])};
    end %properties
    
    
    properties (Hidden, Constant)
        ValidAutoML = [ "hierarchical" "som" "kmeans", ...
            "kmedoids" "gmm" "spectral" ];
    end %properties
    
    
    methods (Access = protected)
        function automl( obj )
            %AUTOML
            
            for i = 1:numel( obj.Trials )
                
                %Get trial/trialline
                trial = obj.Trials(i);
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "automl" );
                
                %Loop over Learners
                models = obj.Learners;

                %Train and Predict
                for iModel = models(:)'
                    try
                        fprintf( "Running: %s\n", iModel )

                        [value, info] = clst.( iModel )(trial.Data, allmodelsettings{:});
                         
                        thisLabel = experiment.Label( 'label', value, 'metadata', info );
 
                        trial.Label = [trial.Label thisLabel] ;
                        
                        result = clst.assignlabel( trial.Data, trial.lastfit().label );
                        
                        trial.Data = result;
                        
                    catch ME
                        
                      obj.handleError( ME, trial )
                      
                    end %try/catch  
                end %for iModel
                
            end %for iTrial
            
        end %function 
        
        %TODO : selectml
        
        function dbscan( obj )
            %DBSCAN
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );

                    %Fit
                    [value, info] = clst.dbscan(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                catch ME
                    
                   obj.handleError( ME, trial )
                   
                end %try
                
            end %for iTrial
            
        end %function
        
        
        function gmm( obj )
            %GMM
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Fit
                    [value, info] = clst.gmm(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
        function hierarchical( obj )
            %HIERARCHICAL
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Fit
                    [value, info] = clst.hierarchical(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
        function kmeans( obj )
            %KMEANS
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );

                    %Fit
                    [value, info] = clst.kmeans(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch
            end %for iTrial
            
        end %function
        
        
        function kmedoids( obj )
            %KMEDOIDS
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Fit
                    [value, info] = clst.kmedoids(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
        function som( obj )
            %SOM Self Organizing Map
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                     %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Fit
                    [value, info] = clst.som(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;
                    
                    
                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
        function spectral( obj )
            %SPECTRAL
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial/trialline
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Fit
                    [value, info] = clst.spectral(trial.Data, allmodelsettings{:});
                    
                    thisLabel = experiment.Label( 'label', value, 'metadata', info );
                    
                    trial.Label = [trial.Label thisLabel] ;
                    
                    %Assign label
                    result = clst.assignlabel( trial.Data, trial.lastfit().label );
                    
                    trial.Data = result;

                catch ME
                   
                    obj.handleError( ME, trial )
                    
                end %try/catch 
            end %for iTrial
            
        end %function
        
        
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
        
        
    end %protected
end %experiment.trial.Unsupervised

