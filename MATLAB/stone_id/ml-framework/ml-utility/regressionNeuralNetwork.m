classdef regressionNeuralNetwork < handle
    %Neural Network Model Container Class
    %
    % Copyright 2021 The MathWorks Inc.
    
    properties (SetAccess = protected, GetAccess = public)
       Name(1,1) string = ""
    end
    
    properties
        Model(1,1)
        PredictorNames(1,:) string
        ResponseName(1,1) string
    end
    
    methods
        function obj = regressionNeuralNetwork(mdl, pnames, rname)
            %Constructor
            
            if nargin > 0               
               obj.Model = mdl;
               obj.PredictorNames = pnames;
               obj.ResponseName = rname;
               obj.Name = mdl.Name;
            end
            
        end %function
        
        function predictions = predict( obj, features )
            
             features = transpose( features );
            
             result = obj.Model( features );         
             predictions = transpose( result );
             
        end %function 
        
    end %methods

end %classdef

