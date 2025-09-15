classdef Record < matlab.mixin.SetGet & ...
        matlab.mixin.Heterogeneous
    %RECORD Summary of this class goes here
    
    properties
        Operation
        Type = "Data"
        Args = [];
    end
    
    methods
        function obj = Record( varargin )
            %FUNCTION Construct an instance of this class
           
            if ~isempty( varargin )
                set( obj, varargin{:} )
            end
        end
        
        function restore( obj, value )
            
            if isempty(obj.Args)
                value.update( obj.Operation )
            else
                value.update( obj.Operation, obj.Args{:} )
            end
            
        end
    end
end %classdef 

