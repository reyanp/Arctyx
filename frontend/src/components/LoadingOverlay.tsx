import { motion } from "framer-motion";

interface LoadingOverlayProps {
  visible?: boolean;
}

export function LoadingOverlay({ visible = true }: LoadingOverlayProps) {
  if (!visible) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#F4E9CD]/95 backdrop-blur-sm"
    >
      <div className="flex flex-col items-center gap-8">
        {/* Spinning Jensen Disc */}
        <img
          src="/jensen-disc.png"
          alt="Loading"
          className="w-48 h-48 animate-slowspin"
        />

        {/* Loading Text */}
        <div className="text-center space-y-2">
          <h3 className="text-xl font-semibold text-foreground">
            Generating Synthetic Data
          </h3>
          <p className="text-sm text-[#468189]">
            This may take a minute... please wait
          </p>
        </div>

        {/* Optional: Progress dots animation */}
        <div className="flex gap-2">
          <motion.div
            className="w-2 h-2 bg-primary rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}
          />
          <motion.div
            className="w-2 h-2 bg-primary rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
          />
          <motion.div
            className="w-2 h-2 bg-primary rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
          />
        </div>
      </div>
    </motion.div>
  );
}

