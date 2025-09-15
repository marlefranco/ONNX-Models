classdef classificationExperiment < matlab.unittest.TestCase
    %test.classificationExperiment classificationExperiment ML unit tests
    %
    %   N.C. Howes, Sudheer Nuggehalli
    %   Copyright 2021 The MathWorks Inc.
    
    properties
        engine
        dataParameters
        modelParameters
        optimizationParameters
    end
    
    properties ( ClassSetupParameter )
        classificationProblem = {"binary", "multiclass"}
    end
    
    properties ( TestParameter )
        dataPipeline = {"NoArgBlockVarargin", "NoArgBlock"}
        dataConfiguration  = {"none", "default"};
        modelConfiguration = {"none", "default"};
        modelSelection     = {"automl", "selectml", "stackml", ...
            "tree", "discr", "nb","knn", "svm", "linear",... 
            "kernel", "ensemble", "ecoc", "nnet", "treebagger"};
        modelSelectionParallel = {"automl", "stackml"};
    end
    
 
    
    methods ( TestClassSetup )
        
        function initialize( testCase, classificationProblem )
            
            %Load Data
            load carbig
            Origin = categorical(cellstr(Origin));
            
            switch classificationProblem
                case "binary"
                    Origin = mergecats(Origin,{'France','Japan','Germany', ...
                        'Sweden','Italy','England'},'NotUSA');
                case "multiclass"
                    Origin = mergecats(Origin,{'France','Germany', ...
                        'Sweden','Italy','England'},'Europe');
            end
            
            data = table(Acceleration,Displacement,Horsepower, ...
                Model_Year,MPG,Weight,Origin);
            
            testCase.engine = data;
            
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
                "Range", {["tree"], ["linear"]},...
                "Type", "Set"); 
            
            modelParameter2 = optimizeParameter.new(...
                "Name", "KFold", ...
                "Range", 5,...
                "Type", "Discrete");
            
            testCase.modelParameters = [...
                modelParameter1
                modelParameter2];
            
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
                modelParameter1
                modelParameter2];
            
        end
        
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'General'} )
        
        function testConstructorDefaults( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}) );
            
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
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}),  ...
                "DataConfiguration", DataConfiguration, ...
                "ModelConfiguration", ModelConfiguration);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
        end
        
        
        function testGuardsPreInitialize( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}) );
            
            %confirm guards pre-initialize (all user defined public methods)
            testCase.helperConfirmGuardsOnMethodsPre( session );
            
        end
        
        
        function testGuardsPostInitialize( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %confirm guards post-initialize (evaluation and io methods)
            testCase.helperConfirmGuardsOnMethodsPost( session );
            
        end
        
        
        function testPipelinesWithDefaults( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %test alternative syntax
            session.run();
            
            session.describe();
            
        end
        
        
        function testPipelinesWithConfiguration( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run pipeline + fit
            fcn = @() session.run();
            testCase.verifyWarningFree(fcn);
            
            %evaluation methods
            session.describe();
            session.sort();
            session.select();
            session.view();
        end
        
        function testPipelineWithNoArgumentBlock( testCase , dataPipeline )
            
            switch dataPipeline
                case "NoArgBlockVarargin"
                    dataFcn = @(x, settings)basePipelineClassificationVarargin(x, settings{:});
                case "NoArgBlock"
                    dataFcn = @(x, settings)basePipelineClassificationNoArgBlock(x, settings{:});
            end
            
            session = experiment.Classification( "Data", testCase.engine, ...
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
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
                "DataConfiguration", dataParameter);
            
            %initialize
            testCase.verifyError( @() session.validate(), char.empty)
            
        end
        
        function testPipelineWithNoCustomProperties( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassificationNoCustomProps(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run prepare
            testCase.verifyWarning( @() session.prepare(), char.empty)
            
        end
        
        function testPipelineWithNoDefinedPartition( testCase )
            
            session = experiment.Classification( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineClassificationNoPartition(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters, ...
                "ModelConfiguration", testCase.modelParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            testCase.verifyWarning( @() session.prepare(), char.empty )
            
        end
        
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Models'} )
        
        function testModelsWithDefaults( testCase, modelSelection )
            
            warning("off", 'stats:bayesoptim:bayesoptim:StandardizeIfOptimizingNBKernelWidth')
            if modelSelection == "selectml" || modelSelection == "stackml"
                
                modelparameter = optimizeParameter.new(...
                    "Name", "HyperparameterOptimizationOptions", ...
                    "Range", { struct(...
                    'MaxObjectiveEvaluations', 1, ...
                    'UseParallel', true, ...
                    'ShowPlots', false, ...
                    'Verbose', 0)},...
                    "Type", "Set");
                
                session = experiment.Classification( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
                    "Model", modelSelection, ...
                    "ModelConfiguration",  modelparameter);
                
            else
                session = experiment.Classification( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
                    "Model", modelSelection);
            end
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %test pipelines, training, running
            session.run();
            
            %evaluation methods
            session.describe();
            warning('on')
            
        end
        
    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Optimization'} )
        
        function testOptimizationWithDefaults( testCase, modelSelection )
                        
            % Don't run optimization tests if MATLAB called from -batch
            % (CI/CD)
            if ~batchStartupOptionUsed
                warning("off", 'stats:bayesoptim:bayesoptim:StandardizeIfOptimizingNBKernelWidth')
                if modelSelection ~= "stackml"
                    session = experiment.Classification( "Data", testCase.engine, ...
                        "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
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
                warning('on')
            end
            
        end
        
    end

    methods (Test)
        function testAutoMLParallel( testCase, modelSelectionParallel )

            % Don't run optimization tests if MATLAB called from -batch
            % (CI/CD)
            if ~batchStartupOptionUsed
                session = experiment.Classification( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineClassification(x, settings{:}), ...
                    "Model", modelSelectionParallel, ...
                    "UseParallel", true);

                %initialize
                session.validate();
                session.build();
                session.preview();

                %test pipelines, training, running
                fcn = @() session.run();
                testCase.verifyWarningFree(fcn);

                session.describe();
            end
        end %function

        function testAutoMLParallelFail( testCase )

            % Don't run optimization tests if MATLAB called from -batch
            % (CI/CD)
            if ~batchStartupOptionUsed

                %Model parameters
                modelParameter1 = optimizeParameter.new(...
                    "Name", "Learners", ...
                    "Range", {["discr", "nb"], ["tree"]},...
                    "Type", "Set");

                modelParameter2 = optimizeParameter.new(...
                    "Name", "KFold", ...
                    "Range", 5,...
                    "Type", "Discrete");

                mParameters = [...
                    modelParameter1
                    modelParameter2];

                session = experiment.Classification( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineClassificationParallelTest(x, settings{:}), ...
                    "Model", "automl", ...
                    "ModelConfiguration", mParameters, ...
                    "UseParallel", true);

                %initialize
                session.validate();
                session.build();
                session.preview();

                %test pipelines, training, running
                warning off
                session.run();
                warning on

                resultTable = session.describe();

                testCase.verifyNotEmpty(resultTable);

                 matchStr = [
                        "Classification (Empty)"
                        "Classification (Empty)"
                        "Classification Tree (fitctree)"];

                testCase.verifyEqual(resultTable.modelType, matchStr)

            end
        end %function

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

