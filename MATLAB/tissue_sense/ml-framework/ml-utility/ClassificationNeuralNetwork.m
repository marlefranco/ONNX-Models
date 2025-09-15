classdef ClassificationNeuralNetwork < NeuralNetwork
    %Neural Network Model Container Class
    
    properties
        Classes
    end
    
    methods
        function obj = ClassificationNeuralNetwork(mdl, pnames, rname, classes)
            %Constructor
            
            if nargin > 0               
               obj.Model = mdl;
               obj.PredictorNames = pnames;
               obj.ResponseName = rname;
               obj.Name = mdl.Name;
               obj.Classes = classes;
            end
            
        end %function
        
        function [predictions, scores] = predict( obj, features )
            
             features = transpose( features );
            
             scores = obj.Model( features );
             value  = scores == max(scores);
             result = transpose( vec2ind(value) );
                
             predictions = categorical(obj.Classes(result), obj.Classes);
             scores = scores';
             
        end %function 
        
    end %methods

end %classdef

