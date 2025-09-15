classdef fits < matlab.unittest.TestCase
    %FITS Test semi supervised models 
    
    properties
        labeled 
        unlabeled 
        unlabeledgtruth
        unseen 
    end
    
    methods (TestClassSetup)

        function initialize( testCase )
            %Generate synthetic dataset with clustering
            
            n = 100;
            dim = 2;
            
            meanvector1 = zeros(dim,1);
            meanvector2 = 10*ones(dim,1);
            
            covariancemat   = eye(dim);

            rng('default'); % for reproducibility
            features = [mvnrnd(meanvector1,covariancemat,n); mvnrnd(meanvector2, covariancemat,n)];
            gtruth = [ones(n,1); 2*ones(n,1)];
           
            test = [mvnrnd(meanvector1,covariancemat,10); mvnrnd(meanvector2, covariancemat,10)];
            
            %Partition into labeled and unlabled set  
            indices = 1:2*n;
            labeledindices =  [128 140 66 131 85 164 7 19 31 14 57 144 107 82 123 194 156 22 54 64];
            unlabeledindices = setdiff(indices, labeledindices); % assume 20 points at random were labeled
            
            testCase.labeled = array2table( ...
                features(labeledindices,:),...
                'VariableNames', "Feature" + (1:size(features,2)) ); % labeled data
            
            testCase.labeled.Response = gtruth(labeledindices); % labels
            
            testCase.unlabeled = array2table( ...
                features(unlabeledindices,:), ...
                'VariableNames', "Feature" + (1:size(features,2)) ); % unlabeled data
            
            testCase.unlabeledgtruth = gtruth( unlabeledindices );
            
            testCase.unseen = array2table( ...
                test, ...
                'VariableNames', "Feature" + (1:size(features,2)) );
            
        end %function 
        
    end %TestClassSetup
    
    methods (Test)
    
        function testGraphFitWithDefaults( testCase )
            
            features = string( testCase.labeled.Properties.VariableNames(1:end-1) );
            response = string( testCase.labeled.Properties.VariableNames(end) );
            gtruth   = testCase.unlabeledgtruth;
            
            [mdl, info] = fits.graph( testCase.labeled, testCase.unlabeled, ...
                "PredictorNames", features, ...
                "ResponseName", response);
            
            disp("graph model accuracy = " + num2str(sum(gtruth==mdl.FittedLabels)/numel(gtruth)*100) + "%")
            
        end %function
        
        function testSelfFitWithDefaults( testCase )
            
            features = string( testCase.labeled.Properties.VariableNames(1:end-1) );
            response = string( testCase.labeled.Properties.VariableNames(end) );
            gtruth   = testCase.unlabeledgtruth;
            
            [mdl, info] = fits.self( testCase.labeled, testCase.unlabeled, ...
                "PredictorNames", features, ...
                "ResponseName", response);
            
            disp("graph model accuracy = " + num2str(sum(gtruth==mdl.FittedLabels)/numel(gtruth)*100) + "%")
           
        end %function
        
        function testGraphPredictWithDefaults( testCase )
            
            features = string( testCase.labeled.Properties.VariableNames(1:end-1) );
            response = string( testCase.labeled.Properties.VariableNames(end) );
            
            [mdl, info] = fits.graph( testCase.labeled, testCase.unlabeled, ...
                "PredictorNames", features, ...
                "ResponseName", response);
            
             predictions = baseml.predict( testCase.unseen, mdl ); 
               
        end
        
        function testSelfPredictWithDefaults( testCase )
            
            features = string( testCase.labeled.Properties.VariableNames(1:end-1) );
            response = string( testCase.labeled.Properties.VariableNames(end) );
            
            [mdl, info] = fits.self( testCase.labeled, testCase.unlabeled, ...
                "PredictorNames", features, ...
                "ResponseName", response);
            
            predictions = baseml.predict( testCase.unseen, mdl );
            
        end
          
    end %Test
      
end %classdef

