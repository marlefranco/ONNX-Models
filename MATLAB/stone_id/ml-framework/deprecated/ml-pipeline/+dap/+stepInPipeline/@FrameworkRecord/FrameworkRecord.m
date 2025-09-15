classdef FrameworkRecord < dap.stepInPipeline.Record 
    %FRAMEWORKRECORD Summary of this class goes here
    
    %   Copyright 2019 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: nhowes $
    %   $Revision: 324 $  $Date: 2019-10-29 13:16:05 -0400 (Tue, 29 Oct 2019) $
    % ---------------------------------------------------------------------
    
    
    properties
        Values 
    end
    
    methods
        function obj = FrameworkRecord( varargin )
            %FRAMEWORKRECORD Construct an instance of this class
       
            obj = obj@dap.stepInPipeline.Record( varargin{:} );
 
        end
        
        function restore(obj, value )
            
            obj.Operation( value )
            
        end
            
    end
end %classdef

