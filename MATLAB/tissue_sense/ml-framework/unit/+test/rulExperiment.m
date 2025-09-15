classdef rulExperiment < matlab.unittest.TestCase
    %test.rulExperiment rulExperiment ML unit tests
    %
    %   N.C. Howes, Sudheer Nuggehalli
    %   MathWorks 2021
    
    
    properties
        pdmEngine
        dataParameters
        modelParameters
        optimizationParameters
    end
    
    properties ( TestParameter )
        dataConfiguration  = {"none", "default"};
        modelConfiguration = {"none", "default"};
        modelSelection     = {"automl", "selectml", ...
            "linDegradation", "expDegradation"};
        optimizationConfiguration = { "light" }
    end
    
    methods ( TestClassSetup )
        
        function initialize( testCase )
            
            %Load data
            load('expTrainTables.mat')
                      
            testCase.pdmEngine = expTrainTables;
            
        end %intialize
        
        function defaultconfigurations( testCase )
            
            % Data parameters
            dataParameter1 = optimizeParameter.new(...
                "Name", "Normalization", ...
                "Range", "zscore",...
                "Type", "Discrete");
            
            testCase.dataParameters = dataParameter1;

            
            %Model parameters
            modelParameter1 = optimizeParameter.new(...
                "Name", "Learners", ...
                "Range", {["linDegradation", "expDegradation"]},...
                "Type", "Set");
            
            modelParameter2 = optimizeParameter.new(...
                "Name", "LifeTimeVariable", ...
                "Range", "Time", ...
                "Type", "Set");
            
            modelParameter3 = optimizeParameter.new(...
                "Name", "DataVariable", ...
                "Range", "Condition", ...
                "Type", "Set");
            
            modelParameter4 = optimizeParameter.new(...
                "Name", "LifeTimeUnit", ...
                "Range", "hours", ...
                "Type", "Set");
            
            testCase.modelParameters = [...
                modelParameter1
                modelParameter2
                modelParameter3
                modelParameter4
                ];          
        end
        
        
        function defaultoptimizationoptions( testCase )
            
            testCase.optimizationParameters = optimizeParameter.new(...
                "Name", "UseParallel", ...
                "Range", [true],...
                "Type", "Discrete");        
            
        end
        
    end %TestClassSetup
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'General'} )
        
        function testConstructorDefaults( testCase )
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
        end
        
        function testConstructorWithConfiguration( testCase, dataConfiguration, modelConfiguration )
            
            switch dataConfiguration
                case "none"
                    DataConfiguration = optimizeParameter.empty(0,1);
                case "default"
                    DataConfiguration = testCase.dataParameters;
            end
            
            switch modelConfiguration
                case "none"
                    ModelConfiguration = optimizeParameter.empty(0,1);
                case "default"
                    ModelConfiguration = testCase.modelParameters;
            end
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}),  ...
                "DataConfiguration", DataConfiguration, ...
                "ModelConfiguration", ModelConfiguration);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
        end
        
        
        function testGuardsPreInitialize( testCase )
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}) );
            
            %confirm guards pre-initialize (all user defined public methods)
            testCase.helperConfirmGuardsOnMethodsPre( session );
            
        end
        
        
        function testGuardsPostInitialize( testCase )
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %confirm guards post-initialize (evaluation and io methods)
            testCase.helperConfirmGuardsOnMethodsPost( session );
            
        end
        
        
        function testPipelinesWithDefaults( testCase )
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %test alternative syntax
            session.run();
            
        end
    
      
        function testPipelinesWithConfiguration( testCase, dataConfiguration, modelConfiguration )
            
            switch dataConfiguration
                case "none"
                    DataConfiguration = optimizeParameter.empty(0,1);
                case "default"
                    DataConfiguration = testCase.dataParameters;
            end
            
            switch modelConfiguration
                case "none"
                    ModelConfiguration = optimizeParameter.empty(0,1);
                case "default"
                    ModelConfiguration = testCase.modelParameters;
            end
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}), ...
                "DataConfiguration", DataConfiguration, ...
                "ModelConfiguration", ModelConfiguration);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run pipeline + fit
            session.run();
            
            %evaluation methods
            session.describe();
            session.sort();
            session.select();
            session.view();
            
        end
   
        function testPipelineWithNoCustomProperties( testCase )
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRULNoCustomProps(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run prepare
            testCase.verifyWarning( @() session.prepare(), char.empty)

            
        end
        
        function testPipelineWithUnknownParameterConfiguration( testCase )
            
            % Data parameters
            dataParameter = optimizeParameter.new(...
                "Name", "Norm", ...
                "Range", ["zscore", "center"],...
                "Type", "Discrete");
            
            session = experiment.RUL( "Data", testCase.pdmEngine, ...
                "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}), ...
                 "DataConfiguration", dataParameter);
            
            %initialize
            testCase.verifyError( @() session.validate(), char.empty) 
            
        end
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Models'} )
        
        function testModelsWithDefaults( testCase, modelSelection )
            
            if modelSelection == "selectml"
                
                modelparameter = optimizeParameter.new(...
                "Name", "UseParallel", ...
                "Range", [true],...
                "Type", "Discrete");  
                
                session = experiment.RUL( "Data", testCase.pdmEngine, ...
                    "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}), ...
                    "Model", modelSelection, ...
                    "ModelConfiguration",  modelparameter);
            else
                
                session = experiment.RUL( "Data", testCase.pdmEngine, ...
                    "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}), ...
                    "Model", modelSelection);
            end
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %test pipelines, training, running
            session.run();
            
        end
           
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Optimization'} )
        
        function testOptimizationWithDefaults( testCase, modelSelection, optimizationConfiguration )
            
            switch optimizationConfiguration
                case "light"
                    ModelConfiguration = testCase.optimizationParameters;
            end
            
             % Don't run optimization tests if MATLAB called from -batch
             % (CI/CD)
             if ~batchStartupOptionUsed
                 session = experiment.RUL( "Data", testCase.pdmEngine, ...
                     "DataFcn", @(x, settings)basePipelineRUL(x, settings{:}), ...
                     "Model", modelSelection, ...
                     "ModelConfiguration", ModelConfiguration);
                 
                 %initialize
                 session.validate();
                 session.build();
                 session.preview();
                 
                 %test pipelines, training, running
                 session.run();
                 
                 %evaluation methods
                 session.describe();
             end
            
        end
        
    end
    
    methods (TestClassTeardown)
        
        function closeFigures(~)
            close("all", "force");
        end
        
    end
    
    methods
        function helperConfirmGuardsOnMethodsPre( testCase, session )
            
            session.prepare();
            session.fit();
            
            testCase.verifyEmpty( session.describe() )
            testCase.verifyEmpty( session.sort() )
            testCase.verifyEmpty( session.select() )
            testCase.verifyEmpty( session.view() )
            
            session.save(1);
            session.plotmetric();
            
        end
        
        
        function helperConfirmGuardsOnMethodsPost( testCase, session )
            
            %Note this helper function omits: prepare, train, run
            
            testCase.verifyEmpty( session.describe() )
            testCase.verifyEmpty( session.sort() )
            testCase.verifyEmpty( session.select() )
            
            session.view();
            session.save(1);
            session.plotmetric();
            
        end
        
    end
    
    
end %classdef

