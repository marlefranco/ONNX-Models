classdef (Hidden) Model < matlab.mixin.SetGet
    %MODEL Summary of this class goes here
    
    properties ( SetAccess = protected, GetAccess = public )
        Name = ""
    end
    
    properties
        mdl
        metadata
        testmetadata
    end
    
    methods
        function obj = Model( varargin )
            %MODEL Construct an instance of this class
  
            if ~isempty( varargin )
               set(obj, varargin{:}) 
            end
        end
    end %public
    
    methods      
        function value = describe( obj  )
        
            meta =cell(numel(obj),1);
            for i = 1:numel(obj)
                
                if ~isempty( obj(i).testmetadata )
                    meta{i} = join(obj(i).metadata, obj(i).testmetadata);
                else
                    meta{i} = obj(i).metadata;
                end
            end
            value = vertcat( meta{:} );
        end
    end
    
    methods (Static)
        function value = default( options )
            
            arguments
               options.ModelType(1,1) string 
               options.Classes (1,:) string  
            end
            
            switch options.ModelType
                
                case "Classification"
                    modelType = "Classification (Empty)";
                    [errorOnResub, errorOnCV, errorOnTest] = deal(nan);
                    
                    padding = num2cell(nan(1,numel(options.Classes)+1));
                    
                    classdefaults = table(padding{:}, ...
                        'VariableNames', [options.Classes(:)' "Avg"]);
                    
                    [precisionOnTrain, recallOnTrain, ...
                        f1ScoreOnTrain, AUCOnTrain] = deal(classdefaults);
                    
                    [precisionOnTest, recallOnTest, ...
                        f1ScoreOnTest, AUCOnTest] = deal(classdefaults);
                    
                    metadata = table(modelType, errorOnResub, errorOnCV, ...
                        precisionOnTrain, recallOnTrain, ...
                        f1ScoreOnTrain, AUCOnTrain);
                    
                    testmetadata = table( modelType, errorOnTest, ...
                        precisionOnTest, recallOnTest, ...
                        f1ScoreOnTest, AUCOnTest );
                    
                    value = experiment.Model( 'metadata', metadata, 'testmetadata', ...
                        testmetadata);
                    
                case "Regression"
                    modelType = "Regression (Empty)";
                    [r2OnResub, r2OnCV, ...
                        mseOnResub, ...
                        rmseOnResub, mseOnCV, ...
                        rmseOnCV, r2OnTest, ...
                        mseOnTest, rmseOnTest] = deal(nan);
                    
                    metadata = table(modelType, r2OnResub, ...
                        r2OnCV, mseOnResub, rmseOnResub, ...
                        mseOnCV, rmseOnCV);
                    
                    testmetadata = table(modelType, r2OnTest, ...
                        mseOnTest, rmseOnTest);
                    
                    value = experiment.Model( 'metadata', metadata, 'testmetadata', ...
                        testmetadata);
                    
                case "PDM"
                    modelType = "RUL (Empty)";
                    [r2OnTrain, rmseOnTrain] = deal(nan);
                    
                    metadata = table(modelType, r2OnTrain, rmseOnTrain);
                    
                    value = experiment.Model( 'metadata', metadata );
                    
                case "SemiSupervised"
                    modelType = "SemiSupervised (Empty)";
                    [errorOnResub, errorOnCV, errorOnTest, ...
                        f1ScoreOnTest, AUCOnTest] = deal(nan);
                    
                    metadata = table(modelType, errorOnResub, errorOnCV);
                    
                    testmetadata = table(modelType,  errorOnTest, ...
                        f1ScoreOnTest, AUCOnTest);
                    
                    value = experiment.Model( 'metadata', metadata, 'testmetadata', ...
                        testmetadata);
                    
                otherwise
                    modelType = "Custom (Empty)";
                    
                    metadata     = table(modelType);
                    testmetadata = table(modelType);
                    
                    value = experiment.Model( 'metadata', metadata, 'testmetadata', ...
                        testmetadata);
                  
            end

        end %function
        
    end %methods
    
end %classdef

