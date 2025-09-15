classdef optimizeContinuous < optimizeParameter
    %optimizeContinuous Summary of this class goes here
    %
    % Example
    %
    %   Parameter = optimizeParameter.new(...
    %      "Name", "Example", ...
    %      "Range", [1 10],...
    %      "Type", "Continuous");
    %   
    %   % Samples     
    %   Parameter.Samples = 10;
    %
    %   % Random   
    %   Parameter.Method = "rand";
    %   Parameter.sample()
    %
    %   % Gridsearch
    %   Parameter.Method = "grid";
    %   Parameter.sample()
    
    
    
    properties ( Access = protected )
        Method_ = "rand";
    end
    
    methods
        function obj = optimizeContinuous( varargin )
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            obj = obj@optimizeParameter( varargin{:} );
            
            try
                validateattributes(obj.Range, {'numeric'}, {'size', [1 2]});
            catch ME
                error('Error.\n%s','When specifying a parameter type of "Continuous", "Range" must be a 2 element vector')
            end
        end
    end
    
    methods
        function set.Method_( obj, value )
            
            validateattributes(value, "string", "scalar");
            
            validTypes = ["grid", "rand"];
            tF = any( ismember(value, validTypes) );
            if tF == false
                error("Invalid Method")
            end
            
            obj.Method_ = value;
        end
    end
    
    methods
        function result = sample(obj)
            
            switch obj.Method
                case "rand"
                    result = (obj.Range(2)-obj.Range(1)).*rand(obj.Samples,1) + obj.Range(1);
                case "grid"
                    result = linspace(obj.Range(1), obj.Range(2), obj.Samples);
            end
              
        end
    end
    
end

