classdef Learner < handle
        %Learner Summary of this class goes here

        % Copyright 2021 The MathWorks Inc.
        
        properties ( SetAccess = immutable )
            Values
        end
        
                
        methods
            function obj = Learner( values )
                obj.Values = values;
            end %function
        end %methods
        
        enumeration

            Classification ([ ...
                "auto"
                "tree"
                "discr"
                "nb"
                "knn"
                "ensemble"
                "ecoc"
                "nnet"
                "svm"
                "linear"
                "kernel"])
            
            Regression ([ ...
                "auto"
                "tree"
                "svm"
                "linear"
                "kernel"
                "gp"
                "ensemble"
                "nnet"
                ])
            
            SemiSupervised ([ ...
                "graph"
                "self"
                ])
            
            Cluster ([ ...
                "kmeans"
                "kmedoids"
                "gmm"
                "spectral"
                "hierarchical"
                "som"
                "dbscan"
                ])
            
            PdM ([ ...
                "expDegradation"
                "linDegradation"
                ])
            
        end %enumeration

        
end %classdef
    
