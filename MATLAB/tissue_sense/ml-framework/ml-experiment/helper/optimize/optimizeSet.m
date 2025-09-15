classdef optimizeSet < optimizeParameter 
    %optimizeSet Summary of this class goes here
    %   Detailed explanation goes here
    
    properties ( Access = protected )
        Method_ = "rand";
    end
    
    
    methods
        function obj = optimizeSet( varargin )
            %UNTITLED8 Construct an instance of this class
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
                    
                    tF = cellfun(@(x)(isstruct(x) | (isobject(x) & ~isstring(x)) ), obj.Range) ;
                    
                    if any(tF)
                        
                        result = obj.Range;
                    else
                        
                        [~, index] = unique(...
                            cellfun(@(x)mat2str(string(x)),obj.Range,'uni',0), ...
                            'stable');
                        
                        result = obj.Range( index );
                    end
                case "rand"
                    result = datasample(obj.Range, obj.Samples);
                
            end
        end
        
    end
end %classdef 

