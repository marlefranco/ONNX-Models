classdef optimizeDiscrete < optimizeParameter
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
   properties ( Access = protected )
        Method_ = "unique";
   end
    
    methods
        function obj = optimizeDiscrete(varargin )
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here

              obj = obj@optimizeParameter( varargin{:} );
            
        end
    end
    
    methods
        function set.Method_( obj, value )

            validateattributes(value, "string", "scalar");
            
            validTypes = ["unique", "rand"];
            tF = any( ismember(value, validTypes) );
            if tF == false
               error("Invalid Method") 
            end
            
            obj.Method_ = value;
        end
    end
    
    methods
    
        function result = sample( obj )
        
            switch obj.Method_
                case "unique"
                    result = unique( obj.Range, 'stable' );
                case "rand"
                    result = datasample(obj.Range, obj.Samples);
                
            end
        end
        
    end
    
end %classdef

