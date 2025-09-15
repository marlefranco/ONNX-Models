classdef clst
    %CLST Cluster analysis for machine learning 
    %
    % clst methods:
    %   kmeans -
    %   kmedoids -
    %   gmm -
    %   spectral -
    %   hierarchical -
    %   som -
    %   dbscan - 
    %   
    
    % MathWorks Consulting 2020
    
    %TODO 
    %Support custom distance metrics 
    
    methods (Static)
           
        function [result, info] = spectral(tbl, custom, options)
            %SPECTRAL
            
            arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Clusters (1,1) double {mustBeInteger} = 4
                options.Distance (1,1) = "euclidean" %TODO support functionhandle & add validation
            end
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Label
            result = spectralcluster( features, custom.Clusters, args{:} );
            result = categorical( result );
            
            %Silhouette score 
            scores = silhouette( features, result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "Spectral clustering (spectralcluster)";
            nClusters  = custom.Clusters;
            
            info = table( modelType, nClusters, avg_Score );
            
        end %function
        
        
        function [result, info, gm] = gmm(tbl, custom, options)
            %GMM
            
            arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Clusters (1,1) double {mustBeInteger} = 4
                custom.Seed (1,1) double {mustBeInteger} = 0
                options.CovarianceType {mustBeMember(options.CovarianceType, [...
                    "full"
                    "diagonal"])} = "full"
                options.SharedCovariance (1,1) logical = true
            end
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Fit 
            rng( custom.Seed )
            gm = fitgmdist( features, custom.Clusters, args{:} );
            
            %Label
            result = gm.cluster( features );
            result = categorical( result );
            
            %Silhouette score 
            scores = silhouette( features, result );
            avg_Score = mean( scores );
             
            %metadata
            modelType  = "Gaussian mixture distribution (fitgmdist)";
            nClusters  = custom.Clusters;
            
            %info = table( modelType, nClusters, gm );
            info = table( modelType, nClusters, avg_Score );
            
        end %clst.fitgmdist
            
        
        function [result, info] = hierarchical(tbl, custom, options)
            %HIERARCHICAL
            
            %TODO support Cutoff 
            
            arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Clusters (1,1)  double {mustBeInteger} = 4;  
                options.Distance (1,1) string ...
                    {mustBeMember(options.Distance,[...
                    "sqeuclidean"
                    "euclidean"
                    "seuclidean"
                    "cityblock"
                    "minkowski"
                    "chebychev"
                    "mahalanobis"
                    "cosine"
                    "correlation"
                    "spearman"
                    "hamming"
                    "jaccard"])} = "euclidean"
            end
            
            options.MaxClust = custom.Clusters;
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Label
            result = clusterdata( features, args{:} );
            result = categorical( result );
            
            %Silhouette score
            scores = silhouette( features, result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "Agglomerative clustering (clusterdata)";
            nClusters  = numel( unique(result) );
            
            info = table( modelType, nClusters, avg_Score );
            
        end %function
        
        
        function [result, info] = kmeans(tbl, custom, options)
            %KMEANS 
            
            arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Clusters (1,1) double {mustBeInteger} = 4
                options.Distance (1,1) string ...
                    {mustBeMember(options.Distance, [...
                    "sqeuclidean"
                    "cityblock"
                    "cosine" 
                    "correlation"
                    "hamming"
                    ])} = "sqeuclidean"
                options.Replicates (1,1) double {mustBeGreaterThan(options.Replicates,0)} = 1;
            end
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Label
            result = kmeans( features, custom.Clusters, args{:} );
            result = categorical( result );
            
            %Silhouette score
            scores = silhouette( features, result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "k-means clustering (kmeans)";
            nClusters  = custom.Clusters;
            
            info = table( modelType, nClusters, avg_Score );
            
        end %function
        
        
        function [result, info] = kmedoids(tbl, custom, options)
            %KMEDIODS  

            arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Clusters (1,1) double {mustBeInteger} = 4
                options.Distance (1,1) string ...
                    {mustBeMember(options.Distance,[...
                    "sqeuclidean"
                    "euclidean"
                    "seuclidean"
                    "cityblock"
                    "minkowski"
                    "chebychev"
                    "mahalanobis"
                    "cosine"
                    "correlation"
                    "spearman"
                    "hamming"
                    "jaccard"])} = "sqeuclidean"
            end
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Label
            result = kmedoids( features, custom.Clusters, args{:} );
            result = categorical( result );
            
            %Silhouette score
            scores = silhouette( features, result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "k-medoids clustering (kmedoids)";
            nClusters  = custom.Clusters;
            
            info = table( modelType, nClusters, avg_Score );
               
        end %function
        
        
        function [result, info] = dbscan(tbl, custom, options)
            %DBSCAN 
            
            arguments
            tbl
            custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)  
            custom.Radius (1,1) double = 1
            custom.MinPts (1,1) {mustBePositive} = 3
            options.Distance  (1,1) string {mustBeMember(options.Distance,[...
                    "precomputed"
                    "euclidean"
                    "squaredeuclidean"
                    "seuclidean"
                    "cityblock"
                    "minkowski"
                    "chebychev"
                    "mahalanobis"
                    "cosine"
                    "correlation"
                    "spearman"
                    "hamming"
                    "jaccard"])} = "euclidean"
            end
            
            %Data
            features   = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args  = namedargs2cell( options );
            
            %Label
            result = dbscan( features, custom.Radius, custom.MinPts, args{:} );
            result = categorical( result );
            
            %Silhouette score
            scores = silhouette( features, result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "Density based clustering (dbscan)";
            nClusters  = numel( unique(result) );
            
            info = table( modelType, nClusters, avg_Score );
            
        end %function
        
        
        function [result, info] = som(tbl, custom, options)
            %SOM Self Organizing Map
            
            arguments
                tbl
                custom.FeatureNames  (1,:) string = string(tbl.Properties.VariableNames)  
                custom.Clusters      (1,:) double = [2 2]; 
                options.CoverSteps   (1,1) double = 100
                options.InitNeighbor (1,1) double = 3
                options.TopologyFcn  (1,1) string ...
                    {mustBeMember(options.TopologyFcn,["hextop", "tritop", "gridtop", "randtop"])} = "hextop";
                options.Distance  (1,1) string ...
                    {mustBeMember(options.Distance,["linkdist", "dist", "mandist", "negdist", "linkdist"])} = "linkdist"    
            end
              
            options.Dimensions = custom.Clusters;
            
            %Data
            features   = transpose( tbl(:, custom.FeatureNames).Variables );

            % Create a Self-Organizing Map
            net = selforgmap( ...
                options.Dimensions , ...
                options.CoverSteps, ...
                options.InitNeighbor, ...
                options.TopologyFcn, ...
                options.Distance ...
                );
            [net, ~]    = train( net, features );
           
            %Label
            y           = net( features );
            result      = transpose( vec2ind(y) );
            result = categorical( result );
            
            
            %Silhouette score
            scores = silhouette( transpose(features), result );
            avg_Score = mean( scores );
            
            %metadata
            modelType  = "Self Organizing Map (som)";
            nClusters  = prod(options.Dimensions, 'all');
            
            info = table( modelType, nClusters, avg_Score );
            
        end %function
        
        
        function result = evalclusters(tbl, custom, options)
            %EVALCLUSTERS
            %TODO
            
             arguments
                tbl
                custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames)
                custom.Algorithm (1,:) string {mustBeMember(custom.Algorithm, [...
                    "kmeans"
                    "linkage"
                    "gmdistribution"])} = "kmeans"
                custom.Criterion (1,:) string {mustBeMember(custom.Criterion, [...
                    "CalinskiHarabasz"
                    "DaviesBouldin"
                    "gap"
                    "silhouette"])} = "CalinskiHarabasz" 
                options.KList (1,:) double {mustBeInteger} = 1:6
                options.Distance (1,1) string ...
                    {mustBeMember(options.Distance, [...
                    "sqeuclidean"
                    "Euclidean"
                    "cityblock"
                    "cosine" 
                    "correlation"
                    "hamming"
                    ])} = "sqeuclidean"
             end
             
             %Data
            features = tbl(:, custom.FeatureNames).Variables;
            
            %Optional Name/Value
            args = namedargs2cell( options );
             
            %Evaluate 
            result = evalclusters( features, custom.Algorithm, custom.Criterion,...
                args{:});
            
        end %function
        
        
        function [result, info] = assignlabel( tbl, labels )
            %ASSIGNLABEL
            
            isTableCol = @(t, thisCol) startsWith(t.Properties.VariableNames, thisCol);
            
            varName = "Label";
            tF = isTableCol( tbl, varName );
            
            name = varName + (sum(tF) + 1);
            
            result = tbl;
            
            result.( name ) = labels;
            
            info = table( name );
            
        end %function
            
        
        function result = rename( labels, newcategories )
            %RENAME 
            
            arguments
               labels categorical
               newcategories cell
            end
            
            result = renamecats(labels, newcategories);
        end %function
        
    end %public
end %classdef

