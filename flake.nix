{
  description = "Markov48 devShell using uv2nix (flake‑parts)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" ];

      perSystem = { pkgs, system, ... }: {
        devShells.default = pkgs.mkShell {
          name = "markov48-dev";
          
          packages = with pkgs; [
            uv
            python312
          ];

          shellHook = ''
			uv venv --python ${pkgs.python312}
			
            echo "✅ Entered Markov48 devShell on ${system}"
            echo "   • Python interpreter = ${pkgs.python312}"
          '';
        };
      };
    };
}
