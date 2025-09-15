classdef optimizeInteger < optimizeParameter
    %optimizeInteger 

    % Copyright 2021 The MathWorks Inc.
    
    properties ( Access = protected )
        Method_ = "rand";
    end
    
    methods
        function obj = optimizeInteger( varargin )
            %optimizeInteger Construct an instance of this class
            obj = obj@optimizeParameter( varargin{:} );
            
            try
                validateattributes(obj.Range, {'numeric'}, {'size', [1 2]});
            catch ME
                error('Error.\n%s','When specifying a parameter type of "Integer", "Range" must be a 2 element vector')
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
                    result = randi(obj.Range,1,obj.Samples);
                case "grid"
                    result = round( linspace(obj.Range(1), obj.Range(2), obj.Samples) );
            end
              
        end
    end
    
end %classdef 

