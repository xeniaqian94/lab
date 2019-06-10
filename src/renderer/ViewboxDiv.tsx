import * as React from "react";
import { getElementScale, getBrowserZoom } from "./geometryFromHtml";
import styled from "styled-components";
import {
  useState,
  useRef,
  useLayoutEffect,
  useEffect,
  useCallback
} from "react";
import interact from "interactjs";
import "@interactjs/types";
import { useDispatch } from "react-redux";
import { iRootState, iDispatch } from "../store/createStore";
import { Box } from "./geometry";
import { useMoveResize } from "./sequenceUtils";
import { MdDeleteForever, MdComment, MdLabel } from "react-icons/md";
import { useNearestSide } from "./geometryFromHtml";
const _AdjustableBox = styled.div`
  position: absolute;
  border: 1px solid green;
  background-color: transparent;
  transition: opacity 0.5s;

  #hoverMenu {
    transition: opacity 0.5s;

    opacity: 0;
    cursor: pointer;
    pointer-events: none;

    #delete:hover {
      transform: scale(1.2);
      color: red;
    }
    #comment:hover {
      transform: scale(1.2);
    }
  }

  &:hover {
    #hoverMenu {
      pointer-events: all;
      opacity: 1;
    }
  }

  &:active {
    #hoverMenu {
      opacity: 0;
    }
  }
`;

type AdjustTypes = "moved" | "resized";
interface AdjustAction {
  type: AdjustTypes;
  payload: { id: string; box: Box };
}

export type MenuTypes = "delete" | "comment" | "scrollToInGraph";
type MenuAction =
  | {
      type: "comment";
      payload: {
        id?: string;
        left?: number;
        top?: number;
        side?: "top" | "bottom";
      };
    }
  | { type: "delete"; payload: { id?: string } };

interface RequiredProps {
  id: "viewbox";
  initBox: Box;
  onChange: (props: AdjustAction | MenuAction) => void;
}
// ViewboxDiv = React.memo(props => {}, shouldMemo)
const shouldMemo = (prevProps: RequiredProps, newProps: RequiredProps) => {
  // fast deep equal?
  const keysToCheck = ["left", "top", "width", "height"];
  for (let key of keysToCheck) {
    if (prevProps.initBox[key] !== newProps.initBox[key]) {
      return false;
    }
  }
  return true;
};
export const AdjustableBox: React.FC<RequiredProps> = React.memo(props => {
  /**
   * Pass in a box from e.g. redux, this will move/resize with a preview, and then emit
   * an event on mouseup, comment too
   */
  const divRef = useRef<HTMLDivElement>(null);
  const { type, payload: box } = useMoveResize(divRef, props.initBox);
  useEffect(() => {
    const payload = { id: props.id, box };
    if (type === "moved") props.onChange({ type: "moved", payload });
    if (type === "resized") props.onChange({ type: "resized", payload });
  }, [type]);

  type onMenuChange = React.ComponentProps<typeof HoverMenu>["onChange"];

  const menuHeight = 20;
  const { side, height } = useNearestSide(divRef) || { side: "", height: 0 };
  const onMenu: onMenuChange = useCallback(
    action => {
      let payload = { id: props.id, ...action.payload };
      if (action.type === "comment") {
        props.onChange({
          type: action.type,
          payload: {
            ...payload,
            side: side as "top" | "bottom"
          }
        });
      } else {
        props.onChange({
          type: action.type,
          payload: {
            ...payload
          }
        });
      }
    },
    [side]
  );
  const top = side === "top" ? -menuHeight : height - 2;

  const { initBox, ...rest } = props;
  const currentBox = { ...initBox, ...box };
  const commentTop =
    side === "top" ? top - currentBox.height / 2 : top + menuHeight;

  const commentBox = {
    top: commentTop,
    left: 0,
    width: currentBox.width,
    height: currentBox.height / 2
  };
  return (
    <_AdjustableBox
      draggable={false}
      id="viewbox"
      ref={divRef}
      style={currentBox}
      {...rest}
      onMouseDown={e => e.stopPropagation()}
      onDragStart={e => e.preventDefault()}
    >
      <HoverMenu
        id="hoverMenu"
        top={top}
        height={menuHeight}
        onChange={onMenu}
      />
    </_AdjustableBox>
  );
}, shouldMemo);

interface HoverMenuProps {
  id: "hoverMenu";
  top: number;
  height: number;
  onChange: (props: MenuAction) => void;
}

const HoverMenu: React.FC<HoverMenuProps> = props => {
  const { top, height, onChange, ...rest } = props;
  return (
    <div
      style={{
        position: "relative",
        top,
        display: "flex",
        justifyContent: "space-around",
        alignContent: "center",
        height
      }}
      {...rest}
    >
      <MdDeleteForever
        size={height - 1}
        id="delete"
        onClick={e => {
          e.stopPropagation();
          onChange({ type: "delete", payload: {} });
        }}
      />
      <MdComment
        size={height - 1}
        id="comment"
        onClick={e => {
          e.stopPropagation();
          onChange({
            type: "comment",
            payload: {
              left: e.clientX,
              top: e.clientY
            }
          });
        }}
      />
    </div>
  );
};
