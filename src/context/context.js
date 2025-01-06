import { createContext, useReducer } from "react";
import initialState from "./initialState";
import mainReducer from "./reducer";

/** Root Reducer for application */
const rootReducer = (state, action) => {
    switch (action.class) {
        default:
            return mainReducer(state, action);
    };
};

/** Context for the application */
const context = createContext();

/**
 * @description Context provider for application
 * @returns {Element}
 */
const Provider = (props) => {
    /** App state and dispatch function */
    const [state, dispatch] = useReducer(rootReducer,initialState);

    /** Context provider component which includes access all context variables */
    return (
        <context.Provider
            value = {{

            }}
        >
            {props.children}
        </context.Provider>
    );
};

export {Provider, context};