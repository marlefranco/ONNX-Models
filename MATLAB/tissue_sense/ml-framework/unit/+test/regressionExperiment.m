classdef regressionExperiment < matlab.unittest.TestCase
    %   test.regressionExperiment regressionExperiment ML unit tests
    %
    %   N.C. Howes, Sudheer Nuggehalli
    %   MathWorks 2021
    
    properties
        engine
        dataParameters
        modelParameters
        optimizationParameters
    end
    
    properties ( TestParameter )
        dataPipeline = {"NoArgBlockVarargin", "NoArgBlock"}
        dataConfiguration  = {"none", "default"};
        modelConfiguration = {"none", "default"};
        modelSelection     = {"automl", "stackml", "selectml", ...
            "tree", "svm", "ensemble", "kernel", "linear", "gp", "nnet"};
    end
    
    methods ( TestClassSetup )
        
        function initialize( testCase )
            
            %Load Data
            load carbig
            
            data = table(Acceleration,Displacement,Horsepower, ...
                Model_Year,Origin,Weight,MPG);
            
            testCase.engine = data(1:5:end, :);
            
        end
    
         function defaultconfigurations( testCase )
            
            % Data parameters
            testCase.dataParameters = optimizeParameter.new(...
                "Name", "Normalization", ...
                "Range", "zscore",...
                "Type", "Discrete");
            
            %Model parameters
            modelParameter1 = optimizeParameter.new(...
                "Name", "Learners", ...
                "Range", {["ensemble"], ["tree"]},...
                "Type", "Set"); %#ok<NBRAK>
            
            modelParameter2 = optimizeParameter.new(...
                "Name", "KFold", ...
                "Range", 5,...
                "Type", "Discrete");
            
            testCase.modelParameters = [...
                modelParameter1
                modelParameter2
                ];
            
        end
        
        
        function defaultoptimizationoptions( testCase )
            
            modelParameter1 = optimizeParameter.new(...
                "Name", "OptimizeHyperparameters", ...
                "Range", "auto",...
                "Type", "Discrete");
            
            modelParameter2 = optimizeParameter.new(...
                "Name", "HyperparameterOptimizationOptions", ...
                "Range", { struct(...
                'MaxObjectiveEvaluations', 1, ...
                'UseParallel', true, ...
                'ShowPlots', false, ...
                'Verbose', 0)},...
                "Type", "Set");
            
            testCase.optimizationParameters = [...
                modelParameter1...
                modelParameter2...
                ];
            
        end
        
    end
    
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'General'} )
        
        function testConstructorDefaults( testCase )
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}) );
            
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
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}),  ...
                "DataConfiguration", DataConfiguration, ...
                "ModelConfiguration", ModelConfiguration);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
        end
        
        
        function testGuardsPreInitialize( testCase )
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}) );
            
            %confirm guards pre-initialize (all user defined public methods)
            testCase.helperConfirmGuardsOnMethodsPre( session );
            
        end
        
        
        function testGuardsPostInitialize( testCase )
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %confirm guards post-initialize (evaluation and io methods)
            testCase.helperConfirmGuardsOnMethodsPost( session );
            
        end
        
        
        function testPipelinesWithDefaults( testCase )
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}) );
            
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
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}), ...
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
        
        function testPipelineWithNoArgumentBlock( testCase , dataPipeline )
            
            switch dataPipeline
                case "NoArgBlockVarargin"
                    dataFcn = @(x, settings)basePipelineRegressionVarargin(x, settings{:});           
                case "NoArgBlock"
                    dataFcn = @(x, settings)basePipelineRegressionNoArgBlock(x, settings{:});
            end
       
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", dataFcn, ...
                "DataConfiguration", testCase.dataParameters);
                        
            testCase.verifyError( @() session.validate(), char.empty )
     
        end
        
        function testPipelineWithUnknownParameterConfiguration( testCase )
            
            % Data parameters
            dataParameter = optimizeParameter.new(...
                "Name", "Norm", ...
                "Range", ["zscore", "center"],...
                "Type", "Discrete");
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}), ...
                 "DataConfiguration", dataParameter);
            
            %initialize
            testCase.verifyError( @() session.validate(), char.empty) 
            
        end
        
        function testPipelineWithNoCustomProperties( testCase )
           
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegressionNoCustomProps(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run prepare
            testCase.verifyWarning( @() session.prepare(), char.empty )

                       
        end
        
        function testPipelineWithNoDefinedPartition( testCase )
            
            session = experiment.Regression( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineRegressionNoPartition(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run pipeline + fit
            testCase.verifyWarning( @() session.prepare(), char.empty )       
            
        end
             
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Models'} ) 
        
        function testModelsWithDefaults( testCase, modelSelection )
           
            if modelSelection == "selectml" || modelSelection == "stackml"
                
                modelparameter = optimizeParameter.new(...
                    "Name", "HyperparameterOptimizationOptions", ...
                    "Range", { struct(...
                    'MaxObjectiveEvaluations', 2, ...
                    'UseParallel', true, ...
                    'ShowPlots', false, ...
                    'Verbose', 0)},...
                    "Type", "Set");
                
                session = experiment.Regression( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}), ...
                    "Model", modelSelection, ...
                    "ModelConfiguration",  modelparameter);
                
            else
                session = experiment.Regression( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}), ...
                    "Model", modelSelection);
            end
            
             %initialize
             session.validate();
             session.build();
             session.preview();
             
             %test pipelines, training, running
             session.run();
             
             session.describe();
        end
        
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Optimization'} )
        
        function testOptimizationWithDefaults( testCase, modelSelection )
            
            % Don't run optimization tests if MATLAB called from -batch
            % (CI/CD)
            if ~batchStartupOptionUsed
                
                if modelSelection ~= "stackml"
                    session = experiment.Regression( "Data", testCase.engine, ...
                        "DataFcn", @(x, settings)basePipelineRegression(x, settings{:}), ...
                        "Model", modelSelection, ...
                        "ModelConfiguration", testCase.optimizationParameters);
                    
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
            
            session.save( 1 );
            session.plotmetric();
            
        end
        
        
        function helperConfirmGuardsOnMethodsPost( testCase, session )
            
            %Note this helper function omits: prepare, train, run
            
            testCase.verifyEmpty( session.describe() )
            testCase.verifyEmpty( session.sort() )
            testCase.verifyEmpty( session.select() )
            
            session.view();
            session.save( 1 );
            session.plotmetric();
            
        end
        
    end
    
end

