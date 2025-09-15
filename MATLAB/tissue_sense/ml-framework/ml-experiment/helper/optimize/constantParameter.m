classdef constantParameter < matlab.mixin.SetGet
    %constantParameter 
    %
    % Syntax:
    %   param = constantParameter( "Name", PipelineVarName, "Value", SelectedValue )
    %
    
    properties
        Name (1,1) string
        Value (1,:)
    end
    
    properties ( Dependent )
        Type
    end
    
    methods
        function obj = constantParameter( varargin )
            %constantParameter Construct an instance of this class

            if nargin > 0
                
                if ~isempty( varargin )
                    set(obj, varargin{:})
                end
                
            end
              
        end %constantParameter
        
        function result = sample( obj )
            %sample
            result = obj.Value;
        end %sample
        
        function result = table( obj )
            %table
            result = table( [obj.Name]', {obj.Value}', ...
                'VariableNames', ["Name" "Value"] );
           
        end
        
    end %public
    
    methods
        function value = get.Type(obj)
            value = string( class(obj) );
        end
    end
    
    methods (Static)
        
        function value = new( options )
            
            arguments
                options.Name (1,1) string
                options.Value (1,:)
            end
            
            value = constantParameter( "Name", options.Name, ...
                "Value", options.Value);
            
        end %new
        
    end %static
    
end %classdef

