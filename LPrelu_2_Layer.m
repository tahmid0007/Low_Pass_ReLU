classdef LPrelu_2_Layer < nnet.layer.Layer
         properties (Learnable)
             A
             B
             Alpha
             Beta
         end
    methods
        function layer = LPrelu_2_Layer(numChannels, name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "LPrelu_2_Layer with " + numChannels + " channels";
            
            layer.A = 5;
            layer.B = 8;
            layer.Alpha = .05;
            layer.Beta = .05/3;
        end
        
        function Z = predict(layer, X)
            
            Z =   (0 < X & X <= 5).*X + (5 < X & X <= 8).*(5 + 0.05.*X) + (8 < X).*(5.4 + 0.02.*X);
            %Z =   (0 < X & X <= layer.A).*X + (layer.A < X & X <= layer.B).*(layer.A + layer.Alpha.*X) + (layer.B < X).*(5.4 + layer.Beta.*X);
        end
        
    end
end
