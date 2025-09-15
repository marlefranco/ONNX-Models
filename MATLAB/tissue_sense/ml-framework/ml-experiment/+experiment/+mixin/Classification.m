classdef Classification < experiment.mixin.Base
    %experiment.mixin.Classification ML Classification methods
    %
    
    properties (Hidden, SetAccess = protected)
        Learners (1,:) string {mustBeMember(Learners, ["tree" "discr" "nb" "knn" "svm" "linear",...
            "kernel" "ensemble" "ecoc" "nnet" "all" "auto" "all-linear" "all-nonlinear" "n/a"])}
    end %properties
    
    
    properties (Hidden, Constant)
        ValidTwoClass = ["tree", "discr", "nb","knn", "svm", "linear",... 
            "kernel", "ensemble", "ecoc", "nnet"];
        
        ValidMultiClass = ["tree", "discr", "nb", "knn",...
            "ensemble", "ecoc", "nnet"]
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
                
                %Train and Predict
                for iModel = models(:)'
                    try
                        
                        fprintf( "Running: %s\n", iModel )
                        [mdl, info]  = fitc.( iModel )(trial.Data, allmodelsettings{:});
                           
                        thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                        trial.model = [trial.model thismodel] ;
                        
                        [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                            trial.lastmodel().metadata.modelType );
                        
                        trial.Data = value;
                        
                        if ~isempty(testinfo)
                            trial.lastmodel().testmetadata = testinfo;
                        end
                        
                    catch ME
                        
                        obj.handleError( ME, trial )
                        
                    end %try/catch
                end %for iModel
                
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
                    
                    [mdl, info]  = fitc.autointernal(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        end %stackml
        
        
        function automl( obj )
            %AUTOML
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "automl" );
                    
                    %Loop over Learners
                    models = obj.Learners;
                    
                    %Train and Predict
                    for iModel = models(:)'
                        try
                            
                            fprintf( "Running: %s\n", iModel )
                            [mdl, info]  = fitc.( iModel )(trial.Data, allmodelsettings{:});
                            
                            thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                            trial.model = [trial.model thismodel] ;
                            
                            [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                                trial.lastmodel().metadata.modelType );
                            
                            trial.Data = value;
                            
                            if ~isempty(testinfo)
                                trial.lastmodel().testmetadata = testinfo;
                            end
                            
                        catch ME
                            
                            obj.handleError( ME, trial )
                            
                        end %try/catch
                    end %for iModel
                    
                catch ME
                    obj.handleError( ME, trial )
                end
                
            end %for iTrial
            
        end %function
        
        
        function selectml( obj )
            %SELECTML
            
            for i = 1:numel( obj.Trials )
                
                %Get item
                trial = obj.Trials(i);
                
                %Initialize model settings
                allmodelsettings = obj.initializeModelParams( trial, "selectml" );
                
                try
                    
                    fprintf( "Running: selectml fitcauto\n")
                    
                    %Version flag
                    ver = str2double(extractBetween(string(version),"R", ("a"|"b")));
                    
                    %Train and predict
                    if  ver >= 2020
                        [mdl, info]  = fitc.auto(trial.Data, allmodelsettings{:});
                    else
                        % Make sure learners are intialized based on 2 vs 3
                        % class valid models
                        obj.validLearners(trial);
                        learnStruct.Learners = obj.Learners;
                        allmodelsettings = [allmodelsettings, namedargs2cell( learnStruct )]; %#ok<AGROW>
                        
                        [mdl, info]  = fitc.autointernal(trial.Data, allmodelsettings{:});
                        
                    end
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %TREE Classification tree model pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train and predict
                    [mdl, info]  = fitc.tree(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function discr( obj )
            %DISCR Discriminant analysis
            
            for i = 1:numel( obj.Trials )
                try                    
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train and predict
                    [mdl, info]  = fitc.discr(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function nb( obj )
            %NB  Naive Bayes classifier pipeline
            
            for i = 1:numel( obj.Trials )
                try                     
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train and predict
                    [mdl, info]  = fitc.nb(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function knn( obj )
            %KNN  KNN classification model
            
            for i = 1:numel( obj.Trials )
                try                     
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train and predict
                    [mdl, info]  = fitc.knn(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %SVM Classification support vector machine model pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Validate binary
                    nClasses = numel( unique(trial.Data.(trial.response) ) );
                    if nClasses > 2
                        return
                        %TODO add warning
                    end
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );
                    
                    %Train and predict
                    [mdl, info]  = fitc.svm(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %LINEAR Linear classification [hd] model pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Validate binary
                    nClasses = numel( unique(trial.Data.(trial.response) ) );
                    if nClasses > 2
                        return
                        %TODO add warning
                    end
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );                    
                    
                    %Train and predict
                    [mdl, info]  = fitc.linear(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %KERNEL Kernel classification model pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Validate binary
                    nClasses = numel( unique(trial.Data.(trial.response) ) );
                    if nClasses > 2
                        return
                        %TODO add warning
                    end
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );

                    %Train and predict
                    [mdl, info]  = fitc.kernel(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %ENSEMBLE Ensemble of classification learners model pipeline
            
            for i = 1:numel( obj.Trials )
                try
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );                    
                    
                    %Train and predict
                    [mdl, info]  = fitc.ensemble(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
        
        
        function ecoc( obj )
            %ECOC Multi-class classification
            
            for i = 1:numel( obj.Trials )
                try       
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );                    
                    
                    %Train and predict
                    [mdl, info]  = fitc.ecoc(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
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
            %Patternnet Multi-class classification
            
            for i = 1:numel( obj.Trials )
                try 
                    %Get item
                    trial = obj.Trials(i);
                    
                    %Initialize model settings
                    allmodelsettings = obj.initializeModelParams( trial, "fit" );                    
                    
                    %Train and predict
                    [mdl, info]  = fitc.nnet(trial.Data, allmodelsettings{:});
                    
                    thismodel = experiment.Model( 'mdl', mdl, 'metadata', info );
                    trial.model = [trial.model thismodel] ;
                    
                    [value, testinfo]  = fitc.predictandupdate( trial.Data, trial.lastmodel().mdl, ...
                        trial.lastmodel().metadata.modelType );
                    
                    trial.Data = value;
                    
                    if ~isempty(testinfo)
                        trial.lastmodel().testmetadata = testinfo;
                    end

                catch ME
                    
                    obj.handleError( ME, trial )
                    
                end  %try/catch
            end %for iTrial
        end %function
        
        
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
    
end %classdef 
