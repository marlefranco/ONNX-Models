classdef Regression < experiment.mixin.Base
    %experiment.mixin.Regression ML Regression methods
    
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
 
        function stackml( obj )
            
            for i = 1:numel( obj.Trials )
                
                %Get trial
                trial = obj.Trials(i);
                
                %Run automl
                allmodelsettings = obj.initializeModelParams( trial, "automl" );
                
                %Loop over Learners
                models = obj.Learners;
                
                %Train and Predict
                for iModel = models(:)'

                     try
                        
                        fprintf( "Running: %s\n", iModel )
                        [mdl, info]  = fitr.( iModel )(trial.Data, allmodelsettings{:});
                           
                        thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                        trial.model = [trial.model thismodel] ;
                        
                        [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                            trial.lastmodel().metadata.modelType );
                        
                        trial.Data = value;
                        
                        if ~isempty(testinfo)
                            trial.lastmodel().testmetadata = testinfo;
                        end
          
                    catch ME
                        
                        obj.handleError( ME, trial )
                        
                    end %try/catch 

                    
                end %for iModel
                
                try
                    
                    %Initialize settings for meta model
                    allmodelsettings = obj.initializeModelParams( trial, "stackml" );
                    
                    %Run meta model
                    fprintf( "Running: stackml\n")
                    [mdl, info]  = fitr.autointernal(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                        trial.lastmodel().metadata.modelType );

                    trial.Data = value;

                    if ~isempty(testinfo)
                        trial.lastmodel().testmetadata = testinfo;
                    end
                    
                    trial.lastmodel().metadata.modelType = trial.lastmodel().metadata.modelType + " Stack";
                    trial.lastmodel().testmetadata.modelType = trial.lastmodel().testmetadata.modelType + " Stack";

                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end %try/catch
            end %for iTrial
            
        end %function
        
        function automl( obj )
            %AUTOML
            
            for i = 1:numel( obj.Trials )
                
                %Get trial
                trial = obj.Trials(i); %trial = trial.result;
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "automl" );
                
                %Loop over Learners
                models = obj.Learners;
                
                %Train and Predict
                for iModel = models(:)'
                    try
                        
                        fprintf( "Running: %s\n", iModel )
                        [mdl, info]  = fitr.( iModel )(trial.Data, allmodelsettings{:});
                           
                        thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                        trial.model = [trial.model thismodel] ;
                        
                        [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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

%         function automl( obj )
%             %AUTOML
%             
%             for i = 1:numel( obj.Trials )
%                 
%                 %Get trial/pipeline
%                 trial = obj.Trials(i); %pipe = trial.result;
%                 
%                 %Initialize model settings
%                 allmodelsettings = obj.initializeModelParams( trial, "automl" );
%                 
%                 %Loop over Learners
%                 models = obj.Learners;
%                                 
%                 if obj.UseParallel
%                     %Train and Predict
%                     
%                     hyperOptFlag = false;
%                     index = zeros(1,length(allmodelsettings));
%                     for ii = 1:2:length(allmodelsettings)-1
%                         index(ii) = strcmp(allmodelsettings{ii}, 'OptimizeHyperparameters');
%                     end
%                     
%                     if any(index)
%                         tF = find(index) + 1;
%                         hyperOptFlag = ~(allmodelsettings{tF(1)} == "none");
%                     end
%                     
%                     % check for hyperparameter optimization model setting
%                     if hyperOptFlag
%                         for iModel = models(:)'
%                             try
%                                 fprintf( "Running: %s\n", iModel )
%                                 obj.trainModels(iModel, trial, allmodelsettings);
%                             catch ME
%                                 
%                                 obj.handleError( ME, trial )
%                                 
%                             end %try/catch
%                         end %for iModel
%                     else
%                         
%                         % Set up execution environment
%                         if isempty(gcp('nocreate'))
%                             parpool()
%                         end
%                         
%                         %Fill futures with parallel tasks
%                         for ii = 1:numel(models)
%                             fprintf( "Queuing: %s\n", models(ii) )
%                             iterations(ii) = parfeval(@obj.trainModels,3,models(ii),trial,allmodelsettings);          %#ok<AGROW>
%                         end 
%                         disp('Models running');
%                         
%                         for ii = 1:numel(models)
%                             
%                             % fetchNext blocks until next results
%                             [~, model, value, info] = fetchNext(iterations);
% 
%                             trial.model = [trial.model model] ;
%                             
%                             trial.Data = value;
%                             
%                             if ~(isempty(trial.model) && isempty(info))
%                                 trial.model(end).testmetadata = info;
%                             end
% 
%                         end
% 
%                     end
%                     
%                 else
%                     
%                     %Train and Predict
%                     for iModel = models(:)'
%                         try
%                             fprintf( "Running: %s\n", iModel )
%                             obj.trainModels(iModel, trial, allmodelsettings);
%                         catch ME
%                             
%                             obj.handleError( ME, pipe )
%                             
%                         end %try/catch
%                     end %for iModel
%                     
%                 end %if UseParallel
%                 
%             end %for iTrial
%             
%         end %function
        
        function selectml( obj )
            %SELECTML
            
            for i = 1:numel( obj.Trials )
                
                %Get trial
                trial = obj.Trials(i); % trial = trial.result;
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "selectml" );
                
                try
                    fprintf( "Running: selectml fitrauto\n")
                    
                    %Version flag
                    ver = str2double(extractBetween(string(version),"R", ("a"|"b")));
                    
                    %Train and predict
                    if ver >= 2020
                        [mdl, info]  = fitr.auto(trial.Data, allmodelsettings{:});
                    else
                        [mdl, info]  = fitr.autointernal(trial.Data, allmodelsettings{:});
                    end
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
              
        function lm( obj )
            %LM Linear regression model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );

                    [mdl, info]  = fitr.lm(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function tree( obj )
            %TREE Regression tree model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );

                    [mdl, info]  = fitr.tree(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function svm( obj )
            %SVM  Regression support vector machine model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.svm(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function gp( obj )
            %GP Gaussian process regression model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.gp(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function ensemble( obj )
            %ENSEMBLE Ensemble of regression learners model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.ensemble(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function linear( obj )
            %LINEAR Linear regression [hd] model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                   %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.linear(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function kernel( obj )
            %KERNEL Kernel regression model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.kernel(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function nnet( obj )
            %KERNEL Kernel regression model trialline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get trial
                    trial = obj.Trials(i); % trial = trial.result;
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    [mdl, info]  = fitr.nnet(trial.Data, allmodelsettings{:});

                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;

                    [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
                trial (1,1) experiment.trial.Supervised
            end %arguments
            
            warning(ME.identifier, '%s', ME.message)
            
            mdl = experiment.Model.default( "ModelType", obj.Type );
            
            errorstack = string( {ME.stack.name} );
            
            if isempty(trial.model)
                trial.model = mdl;
            elseif any( contains(errorstack, "fitr.predictandupdate") )
                mdl.testmetadata.modelType = trial.model(end).metadata.modelType;
                trial.model(end).testmetadata = mdl.testmetadata;
            else %Everything else... model or generic error
                trial.model = [trial.model mdl];
            end %if isempty(trial.Model)
            
        end %function
       

    end %methods

    methods (Static, Access = private)
%         function [mdl, value, info] = trainModels(iModel, trial, allmodelsettings, mthd)
%             
%             arguments
%                 
%                 
%             end
%             
%             switch
%             [mdl, info]  = fitr.( iModel )(trial.Data, allmodelsettings{:});
%             
%             
%             end
%             
%             thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
%             trial.model = [trial.model thismodel] ;
%             
%             [value, testinfo]  = fitr.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
%                 trial.lastmodel().metadata.modelType );
%             
%             trial.Data = value;
%             
%             if ~isempty(testinfo)
%                 trial.lastmodel().testmetadata = testinfo;
%             end
%             
%         end
% 
    end % static methods
        
end %classdef

% %Version flag
% ver = extractBetween(string(version),"(",")");
% if ver == "R2020b"
%     modelFunction = @(x, settings)fitr.auto(x, settings{:});
% else
%     modelFunction = @(x, settings)fitr.autointernal(x, settings{:});
% end