declare module "react-cytoscapejs" {
  import type { CSSProperties, ComponentType } from "react";
  import type { Core, ElementDefinition, LayoutOptions } from "cytoscape";

  export interface CytoscapeComponentProps {
    elements?: ElementDefinition[] | Record<string, unknown>[];
    layout?: LayoutOptions | Record<string, unknown>;
    stylesheet?: unknown[];
    style?: CSSProperties;
    cy?: (cy: Core) => void;
    className?: string;
    maxZoom?: number;
    minZoom?: number;
    wheelSensitivity?: number;
    boxSelectionEnabled?: boolean;
    autoungrabify?: boolean;
    autounselectify?: boolean;
    [key: string]: unknown;
  }

  const CytoscapeComponent: ComponentType<CytoscapeComponentProps>;
  export default CytoscapeComponent;
}

declare module "cytoscape-fcose" {
  import type { Ext } from "cytoscape";
  const extension: Ext;
  export default extension;
}
