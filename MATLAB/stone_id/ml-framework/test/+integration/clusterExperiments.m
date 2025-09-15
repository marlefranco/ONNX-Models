classdef clusterExperiments < matlab.unittest.TestCase
    %test.clusterExpriments clusterExpriments ML unit tests
    %
    %   N.C. Howes, Sudheer Nuggehalli
    %   Copyright 2021 The MathWorks Inc.
    
    
    properties
        engine
        dataParameters
        modelParameters
    end
    
    properties ( TestParameter )
        dataConfiguration  = {"none", "default"};
        modelConfiguration = {"none", "default"};
        modelSelection     = {"automl", "hierarchical", "som", "kmeans", ...
            "kmedoids", "gmm", "spectral"};
    end
    
    methods ( TestClassSetup )
        
        function initialize( testCase )
            
            %Load data
            rng default; % For reproducibility
            X = [randn(100,2)*0.75+ones(100,2);
                    randn(100,2)*0.5-ones(100,2)];
                      
            testCase.engine = array2table(X, ...
                "VariableNames", ["Feature 1", "Feature 2"]);
            
        end %intialize
        
        function defaultconfigurations( testCase )
            
            % Data parameters
            testCase.dataParameters = optimizeParameter.new(...
                "Name", "Normalization", ...
                "Range", ["zscore", "center"],...
                "Type", "Discrete");
                 
            %Model parameters
            testCase.modelParameters = optimizeParameter.new(...
                "Name", "Learners", ...
                "Range", {["hierarchical", "som", "kmeans"], ["kmedoids"]}, ...
                "Type", "Set");
            
        end
        
    end %TestClassSetup
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'General'} )
        
        function testConstructorDefaults( testCase )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}) );
            
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
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}),  ...
                "DataConfiguration", DataConfiguration, ...
                "ModelConfiguration", ModelConfiguration);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
        end
        
        
        function testGuardsPreInitialize( testCase )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}) );
            
            %confirm guards pre-initialize (all user defined public methods)
            testCase.helperConfirmGuardsOnMethodsPre( session );
            
        end
        
        
        function testGuardsPostInitialize( testCase )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}) );
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %confirm guards post-initialize (evaluation and io methods)
            testCase.helperConfirmGuardsOnMethodsPost( session );
            
        end
        
        
        function testPipelinesWithDefaults( testCase )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}) );
            
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
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}), ...
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
        
        function testSelectMethod( testCase )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}), ...
                "DataConfiguration", testCase.dataParameters);
            
            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %run pipeline + fit
            session.run();
            
            %select
            testCase.verifyClass(session.select(), "table");
            testCase.verifyError( @() session.select(1, "Data", "original" ), "key:KeyNotEmpty" )
            testCase.verifyClass(session.select( "Data", "" ), "categorical")
            testCase.verifyClass(session.select( "Data", "original", "Key", "Feature 1"), "table")
            testCase.verifyClass(session.select( "Key", "Feature 1"), "table")
            
        end

    end
    
    methods ( Test, ParameterCombination = 'pairwise', TestTags = {'Models'} )
        
        function testModelsWithDefaults( testCase, modelSelection )
            
            session = experiment.Cluster( "Data", testCase.engine, ...
                "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}), ...
                "Model", modelSelection);

            %initialize
            session.validate();
            session.build();
            session.preview();
            
            %test pipelines, training, running
            session.run();
            
        end
           
    end

    methods (Test)
        function testAutoMLParallel( testCase )

            % Don't run optimization tests if MATLAB called from -batch
            % (CI/CD)
            if ~batchStartupOptionUsed
                session = experiment.Cluster( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineCluster(x, settings{:}), ...
                    "Model", "automl", ...
                    "UseParallel", true);

                %initialize
                session.validate();
                session.build();
                session.preview();

                %test pipelines, training, running
                fcn = @() session.run();
                testCase.verifyWarningFree(fcn)

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
                    "Range", {["hierarchical", "kmeans"], ["kmedoids"]},...
                    "Type", "Set");

                mParameters = [...
                    modelParameter1];

                session = experiment.Cluster( "Data", testCase.engine, ...
                    "DataFcn", @(x, settings)basePipelineClusterParallelTest(x, settings{:}), ...
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
                        "Cluster Empty"
                        "Cluster Empty"
                        "Cluster Empty"];

                testCase.verifyEqual(resultTable.modelType, matchStr)

            end
        end %function
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
    
    
    
end